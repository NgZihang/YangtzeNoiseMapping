import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold # Import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import Linear, PReLU, Sequential
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import SAGEConv

import warnings
# Suppress specific UserWarnings from sklearn if they are not critical
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
# Suppress specific UserWarnings from PyTorch related to DataLoader num_workers on Windows
warnings.filterwarnings("ignore", category=UserWarning, message=".*The DataLoader found that .* num_workers .*")


# --- Configuration ---
INPUT_FILE = 'DummyInput.csv' # Your input file name
TARGET_COLUMN = 'corrected_final_total_dBA'

# Features for each SHIP node
# IMPORTANT: Ensure these columns exist in your INPUT_FILE
SHIP_NODE_FEATURES = ['distance_to_test_point_km', 'ship_s_interpolated', 'ship_orig_l', 'cos'] 

# Features for the HYDROPHONE node (or global graph features)
# IMPORTANT: Ensure these columns exist in your INPUT_FILE if uncommented
HYDROPHONE_NODE_FEATURES = []

# If we use distance as an edge feature (SAGEConv doesn't use this directly)
EDGE_FEATURES = ['distanceW'] # Illustrative for Data object

# Model Hyperparameters
HIDDEN_CHANNELS = 512
NUM_GNN_LAYERS = 3 # Number of SAGEConv layers
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.4
NUM_EPOCHS_PER_FOLD = 2001 # Number of epochs to run for each fold
BATCH_SIZE = 64
K_FOLDS = 10 # Number of folds for cross-validation

# Model Saving Configuration
SAVE_MODEL_DIR = 'saved_models_gnn'
# Track the best training MSE encountered across all epochs in a fold for saving
BEST_TRAIN_MSE_FOR_FOLD = float('inf')



# --- PyG Dataset Class ---
class ShipNoiseDataset(Dataset):
    def __init__(self, root, df, ship_features, hydrophone_features, target_col, scalers=None, fit_scalers=False, transform=None, pre_transform=None):
        # Ensure root directory exists for saving processed data if pre_transform is used
        if root is not None and not os.path.exists(root):
             os.makedirs(root)
             
        self.df = df
        self.ship_features_cols = ship_features
        self.hydrophone_features_cols = hydrophone_features
        self.target_col = target_col
        self.grouping_keys = ['timestamp_utc8', 'lat', 'lon']
        self.scalers = scalers if scalers else {}
        self.fit_scalers = fit_scalers

        # Fit scalers only if requested (happens in the first dataset creation for a fold)
        if self.fit_scalers:
            # Scale ship features
            # Need to collect all ship data *across all groups in this df*
            all_ship_data_for_scaler = self.df[self.df['ship_id'] != -1].copy()
            if not all_ship_data_for_scaler.empty and self.ship_features_cols:
                self.scalers['ship_features'] = StandardScaler()
                self.scalers['ship_features'].fit(all_ship_data_for_scaler[self.ship_features_cols])
                print(f"Fitted ship_features scaler on {len(all_ship_data_for_scaler)} rows.")
            else:
                self.scalers['ship_features'] = None
                print("No ship data found or no ship features defined to fit scaler.")

            # Scale hydrophone features
            # Take the first row of each group for hydrophone features
            hydro_df_features_to_scale = self.df.groupby(self.grouping_keys).first().reset_index()
            # Add lat, lon to hydrophone features list for scaling if desired
            h_features_list = [f for f in self.hydrophone_features_cols if f in hydro_df_features_to_scale.columns]

            if h_features_list:
                 # Select only the columns present in the DataFrame before fitting
                 h_features_to_fit = hydro_df_features_to_scale[h_features_list].copy()
                 # Drop rows where any of these critical hydro features might be NaN if not filled
                 h_features_to_fit.dropna(subset=h_features_list, inplace=True)
                 
                 if not h_features_to_fit.empty:
                     self.scalers['hydrophone_features'] = StandardScaler()
                     self.scalers['hydrophone_features'].fit(h_features_to_fit)
                     print(f"Fitted hydrophone_features scaler on {len(h_features_to_fit)} rows.")
                 else:
                     self.scalers['hydrophone_features'] = None
                     print("No hydrophone data found or hydrophone features defined to fit scaler.")
            else:
                self.scalers['hydrophone_features'] = None
                print("No hydrophone features defined to fit scaler.")

        # Ensure self.scalers['ship_features'] and self.scalers['hydrophone_features'] exist even if None
        self.scalers['ship_features'] = self.scalers.get('ship_features')
        self.scalers['hydrophone_features'] = self.scalers.get('hydrophone_features')


        self.processed_data_list = self._process_data()
        # print(f"Created dataset with {len(self.processed_data_list)} graphs.")
        super().__init__(root, transform, pre_transform) # Call super last if processing in __init__

    def _process_data(self):
        data_list = []
        grouped = self.df.groupby(self.grouping_keys)
        
        
        for name, group in grouped: # tqdm(grouped, desc="Processing timestamp groups"): # Removed tqdm here to avoid nested bars
            # Hydrophone node features
            hydro_lat, hydro_lon = name[1], name[2]
            
            # hydro_node_f = []
            first_row_group = group.iloc[0]

            hydro_node_f_list = []
            first_row_group = group.iloc[0]
            present_hydro_cols = [f for f in self.hydrophone_features_cols if f in first_row_group]

            if self.scalers.get('hydrophone_features'):
                 # The scaler was fitted on present_hydro_cols + ['lat', 'lon']
                 features_for_scaler = ([first_row_group[f] for f in present_hydro_cols] if present_hydro_cols else [])
                 scaled_features = self.scalers['hydrophone_features'].transform(np.array(features_for_scaler).reshape(1, -1))[0]
                 hydro_node_f_list.extend([0, 0, 0, 0])

            else:
                hydro_node_f_list.extend([0, 0, 0, 0])


            hydro_node_features = torch.tensor([hydro_node_f_list], dtype=torch.float)


            # Ship node features
            ship_group = group[group['ship_id'] != -1].copy() # Filter out placeholder rows
            
            if not ship_group.empty and self.ship_features_cols:
                # Ensure columns exist before accessing
                present_ship_cols = [f for f in self.ship_features_cols if f in ship_group.columns]
                if present_ship_cols:
                    ship_node_f_df = ship_group[present_ship_cols].astype(np.float32)
                    if self.scalers.get('ship_features'):
                        ship_nodes_features = torch.tensor(self.scalers['ship_features'].transform(ship_node_f_df), dtype=torch.float)
                    else:
                        ship_nodes_features = torch.tensor(ship_node_f_df.values, dtype=torch.float)
                else: # No ship features were actually found in the columns
                     ship_nodes_features = torch.empty((len(ship_group), 0), dtype=torch.float) # Empty tensor
                     print(f"Warning: No configured SHIP_NODE_FEATURES found in DataFrame for group {name}. Creating empty ship feature tensor.")

                num_ships = len(ship_nodes_features)
                
                # Combine hydrophone (node 0) and ship features (nodes 1 to N)
                # Handle case where ship_nodes_features is empty
                if num_ships > 0:
                     x = torch.cat([hydro_node_features, ship_nodes_features], dim=0)
                else: # No ships, just hydrophone node
                     x = hydro_node_features


                # Edge index: ships connect to hydrophone (node 0)
                edge_index = torch.empty((2,0), dtype=torch.long) # Default to no edges
                if num_ships > 0:
                    ship_indices = torch.arange(1, num_ships + 1, dtype=torch.long)
                    hydro_indices = torch.zeros(num_ships, dtype=torch.long)
                    edge_index_to_hydro = torch.stack([ship_indices, hydro_indices], dim=0)
                    edge_index_from_hydro = torch.stack([hydro_indices, ship_indices], dim=0)
                    edge_index = torch.cat([edge_index_to_hydro, edge_index_from_hydro], dim=1)

                # Edge features (e.g., distanceW)
                edge_attr = None
                if EDGE_FEATURES:
                     present_edge_cols = [f for f in EDGE_FEATURES if f in ship_group.columns]
                     if present_edge_cols and num_ships > 0:
                         # Assuming the first edge feature is distanceW and is symmetric
                         edge_attr_vals = torch.tensor(ship_group[present_edge_cols].values, dtype=torch.float)
                         # Repeat for both directions
                         edge_attr = torch.cat([edge_attr_vals, edge_attr_vals], dim=0)
                         # Note: Need to handle scaling for edge_attr if necessary, similar to node features.
                         # For simplicity here, assuming edge features are already scaled or don't need scaling.
                     elif EDGE_FEATURES and num_ships == 0:
                          # Need empty edge_attr tensor if EDGE_FEATURES is defined but no ships
                          edge_attr = torch.empty((0, len(present_edge_cols) if present_edge_cols else 0), dtype=torch.float)


            else: # No ships in this group
                x = hydro_node_features # Only hydrophone node
                edge_index = torch.empty((2,0), dtype=torch.long) # No edges
                edge_attr = torch.empty((0, len(EDGE_FEATURES) if EDGE_FEATURES else 0), dtype=torch.float) # Empty edge attr
                num_ships = 0

            # Target value
            # Ensure target column exists and is not NaN for this group
            if self.target_col in group.columns and pd.notna(group[self.target_col].iloc[0]):
                 y = torch.tensor([group[self.target_col].iloc[0]], dtype=torch.float)
                 data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_hydrophone_nodes=1, num_ship_nodes=num_ships))
            else:
                 # print(f"Warning: Skipping group {name} due to missing or NaN target value.")
                 pass # Skip graph if target is missing or NaN
            
        return data_list

    def len(self):
        return len(self.processed_data_list)

    def get(self, idx):
        return self.processed_data_list[idx]

# --- GNN Model ---
class GNNNoisePredictor(torch.nn.Module):
    def __init__(self, num_hydrophone_features, num_ship_features, num_edge_features, hidden_channels, num_gnn_layers, out_channels=1):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers

        # Initial linear layers to project to hidden_channels
        self.hydro_lin = Linear(num_hydrophone_features, hidden_channels, bias=False)
        # Create ship_lin only if there are potential ship features
        self.ship_lin = Linear(num_ship_features, hidden_channels) if num_ship_features > 0 else None

        # GNN Layers (using SAGEConv)
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_gnn_layers):
             # SAGEConv does not directly use edge_attr in its basic form
             self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels))

        # Readout and final MLP
        # Output from the hydrophone node embedding
        self.out_mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            PReLU(), # Using PReLU as per user's code
            Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.batch, data.edge_attr
        
        # Apply initial linear projections
        processed_x_list = []
        current_node_idx_in_batch = 0
        for i in range(data.num_graphs): # Iterate through graphs in the batch
            # Find the global indices for nodes of the current graph
            start_idx = data.ptr[i]
            end_idx = data.ptr[i+1]
            graph_x = x[start_idx:end_idx]

            # Project hydrophone node (always the first node of the graph in the processed Data object)
            hydro_x_proj = self.hydro_lin(graph_x[0:1])
            processed_x_list.append(hydro_x_proj)

            # Project ship nodes if they exist and ship_lin is defined
            if graph_x.size(0) > 1 and self.ship_lin is not None:
                ship_x_proj = self.ship_lin(graph_x[1:])
                processed_x_list.append(ship_x_proj)
            elif graph_x.size(0) > 1 and self.ship_lin is None:
                 pass # No ship projection if ship_lin is None

        h = torch.cat(processed_x_list, dim=0) # Concatenate projected features back


        # Apply GNN layers
        for i in range(self.num_gnn_layers):
            h = F.dropout(h, p=0.3, training=self.training)
            h = F.relu(h)
            h = self.conv_layers[i](h, edge_index) # SAGEConv doesn't use edge_attr here


        # No activation/dropout after the last GNN layer before the readout

        # Readout: Extract hydrophone node embeddings
        hydrophone_node_indices_in_batch = data.ptr[:-1] # These are the global indices of node 0 for each graph in the batch
        hydrophone_embeddings = h[hydrophone_node_indices_in_batch]
        
        # Pass hydrophone embeddings through output MLP
        out = self.out_mlp(hydrophone_embeddings)
        return out


def train_fold(model, loader, optimizer, scheduler, criterion, device, fold_idx, best_train_mse_for_fold_ref, epoch):
    """Trains the model for one fold and saves best model based on train MSE."""
    model.train()
    total_loss = 0
    num_graphs = 0
    
    # Use tqdm for the epoch loop outside this function
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        num_graphs += data.num_graphs

    avg_train_loss = total_loss / num_graphs if num_graphs > 0 else float('inf')

    # Check if this is the best train MSE so far for this fold
    global BEST_TRAIN_MSE_FOR_FOLD # Access the global variable
    
    if avg_train_loss < BEST_TRAIN_MSE_FOR_FOLD and epoch > 800:
        BEST_TRAIN_MSE_FOR_FOLD = avg_train_loss
        # Save the model
        # Create directory if it doesn't exist
        if not os.path.exists(SAVE_MODEL_DIR):
            os.makedirs(SAVE_MODEL_DIR)
            
        model_save_path = os.path.join(SAVE_MODEL_DIR, f'best_train_mse_model_fold_{fold_idx+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        # print(f" --> Saved best model for Fold {fold_idx+1} at Epoch {current_epoch} with Train MSE: {avg_train_loss:.4f}") # Need epoch info


    scheduler.step() # Step scheduler after each epoch (or batch, depending on scheduler)
    
    return avg_train_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates the model on a given loader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_ys = []
    num_graphs = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out.squeeze(), data.y.squeeze())
        total_loss += loss.item() * data.num_graphs
        num_graphs += data.num_graphs
        # 处理预测值和标签，确保它们至少有一个维度
        pred = out.squeeze().cpu()
        y = data.y.squeeze().cpu()
        
        # 如果预测值或标签是零维张量，扩展维度
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
            
        all_preds.append(pred)
        all_ys.append(y)
    
    avg_loss = total_loss / num_graphs if num_graphs > 0 else float('nan')
    # print(all_preds)
    if num_graphs > 0:
        preds_tensor = torch.cat(all_preds)
        ys_tensor = torch.cat(all_ys)
        mse = mean_squared_error(ys_tensor.numpy(), preds_tensor.numpy())
        mae = mean_absolute_error(ys_tensor.numpy(), preds_tensor.numpy())
        r2 = r2_score(ys_tensor.numpy(), preds_tensor.numpy())
    else:
        mse, mae, r2 = float('nan'), float('nan'), float('nan')
        preds_tensor, ys_tensor = np.array([]), np.array([]) # Return empty numpy arrays


    return avg_loss, mse, mae, r2, ys_tensor.numpy(), preds_tensor.numpy()


# --- Main Execution with K-Fold ---
def main_kfold_gnn():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading {INPUT_FILE}...")
    try:
        df_raw = pd.read_csv(INPUT_FILE)
        
        # --- Feature Engineering (as in user's updated code) ---
        # Ensure these columns exist or handle errors/NaNs
        df_raw['cos'] = np.cos((df_raw['ball_direction_deg'].fillna(0).to_numpy() - df_raw['ship_orig_c'].fillna(0).to_numpy()) / 360.0 * np.pi)
        df_raw['distanceW'] = 1.0 / df_raw['distance_to_test_point_km'] # This can stay global, maybe scale later

        # Handle NaNs in newly engineered features before they cause issues
        engineered_cols = ['cos', 'distanceW'] # Add other engineered cols here
        for col in engineered_cols:
             if col in df_raw.columns:
                  df_raw[col] = df_raw[col].replace([np.inf, -np.inf], np.nan).fillna(df_raw[col].median() if df_raw[col].median() is not np.nan else 0) # Replace inf with NaN, then fill NaN with median or 0
                  if df_raw[col].std() == 0: # Handle case where std is zero after filling
                       print(f"Warning: Standard deviation of engineered column '{col}' is zero after filling. Scaling might be problematic.")


        # Ensure all configured SHIP_NODE_FEATURES and HYDROPHONE_NODE_FEATURES columns exist
        required_cols = SHIP_NODE_FEATURES + HYDROPHONE_NODE_FEATURES + [TARGET_COLUMN, 'timestamp_utc8', 'lat', 'lon', 'ship_id'] + engineered_cols
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
             raise ValueError(f"Missing required columns in {INPUT_FILE}: {missing_cols}")
        
        cols_to_fill_nan = list(set(SHIP_NODE_FEATURES + HYDROPHONE_NODE_FEATURES + EDGE_FEATURES + ['lat', 'lon'])) # Ensure lat/lon are included if scaled
        for col in cols_to_fill_nan:
            if col in df_raw.columns:
                 # Simple fill with median for numerical columns
                 if pd.api.types.is_numeric_dtype(df_raw[col]):
                      df_raw[col] = df_raw[col].fillna(df_raw[col].median() if df_raw[col].median() is not np.nan else 0)
                 # Add other dtypes handling if needed
            else:
                 print(f"Warning: Column '{col}' specified in feature lists not found in DataFrame for NaN filling.")


    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        # print("Please ensure the input file exists or uncomment/fix the dummy data generator.")
        return
    except Exception as e:
        print(f"Error loading or processing {INPUT_FILE}: {e}")
        return
        
    if df_raw.empty:
        print("DataFrame is empty after loading/filtering.")
        return

    # Get unique events for K-Fold splitting
    grouping_keys = ['timestamp_utc8', 'lat', 'lon']
    unique_events = df_raw[grouping_keys].drop_duplicates().reset_index(drop=True)
    
    if len(unique_events) < K_FOLDS:
        print(f"Not enough unique events ({len(unique_events)}) for {K_FOLDS}-fold cross-validation. Need at least {K_FOLDS}.")
        return

    # Setup KFold
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=43) # Use consistent random_state for reproducibility

    # Store results for each fold
    fold_results = []
    all_test_preds = []
    all_test_actuals = []



    hydro_cols_for_dataset = [f for f in HYDROPHONE_NODE_FEATURES if f in df_raw.columns]

    print("Determining feature dimensions from a sample dataset...")
    try:
        # Create a small temporary dataset just to get feature dimensions
        temp_df = df_raw.copy() # Use a subset for speed
        if temp_df.empty: # Handle case where df_raw has fewer than 100 rows
             temp_df = df_raw.copy()
             if temp_df.empty:
                  raise ValueError("DataFrame is empty, cannot determine feature dimensions.")

        # Need a dummy root for the temporary dataset
        dummy_root = 'temp_data_check_dims'
        if not os.path.exists(dummy_root):
             os.makedirs(dummy_root)

        # Fit scalers on this temporary data to get the scaled dimensions
        temp_dataset = ShipNoiseDataset(root=dummy_root, df=temp_df,
                                        ship_features=SHIP_NODE_FEATURES,
                                        hydrophone_features=hydro_cols_for_dataset,
                                        target_col=TARGET_COLUMN,
                                        fit_scalers=True)
        
        if not temp_dataset.processed_data_list:
             raise ValueError("Dummy dataset processing failed, no graphs created to determine dimensions.")

        # Get dimensions from the first processed graph
        sample_graph = temp_dataset.processed_data_list[0]
        
        # The first node is always hydrophone
        sample_hydro_features_dim = sample_graph.x[0].size(0)
        # Check if there are ship nodes (if any ship features defined)
        sample_ship_features_dim = 0
        if SHIP_NODE_FEATURES and sample_graph.x.size(0) > 1:
             sample_ship_features_dim = sample_graph.x[1].size(0)
        
        # Determine edge feature dimension (if any edge features defined)
        sample_edge_features_dim = sample_graph.edge_attr.size(1) if sample_graph.edge_attr is not None and sample_graph.edge_attr.size(0) > 0 else 0

        print(f"Inferred Hydrophone Feature Dimension: {sample_hydro_features_dim}")
        print(f"Inferred Ship Feature Dimension: {sample_ship_features_dim}")
        print(f"Inferred Edge Feature Dimension: {sample_edge_features_dim}")


        print(f"Temporary directory '{dummy_root}' created. You can delete it manually.")

    except Exception as e:
         print(f"Error determining feature dimensions: {e}")
         # Fallback: try to estimate based on configuration lists + lat/lon
         print("Attempting to estimate feature dimensions based on configuration...")
         num_model_hydro_features = (len([f for f in HYDROPHONE_NODE_FEATURES if f in df_raw.columns]) if HYDROPHONE_NODE_FEATURES else 0) + 2 # Assuming lat/lon always added
         num_model_ship_features = len([f for f in SHIP_NODE_FEATURES if f in df_raw.columns]) if SHIP_NODE_FEATURES else 0
         sample_edge_features_dim = len([f for f in EDGE_FEATURES if f in df_raw.columns]) if EDGE_FEATURES else 0
         print(f"Estimated Hydrophone Feature Dimension: {num_model_hydro_features}")
         print(f"Estimated Ship Feature Dimension: {num_model_ship_features}")
         print(f"Estimated Edge Feature Dimension: {sample_edge_features_dim}")
         print("Warning: Using estimated dimensions. Model might fail if dimensions are incorrect.")
         sample_hydro_features_dim = num_model_hydro_features # Use estimated for model init
         sample_ship_features_dim = num_model_ship_features # Use estimated for model init


    print(f"\nStarting {K_FOLDS}-fold cross-validation...")

    for fold_idx, (train_event_indices, val_event_indices) in enumerate(kf.split(unique_events)):
        print(f"\n--- Starting Fold {fold_idx+1}/{K_FOLDS} ---")

        # Get event identifiers for the current fold
        train_events = unique_events.iloc[train_event_indices]
        val_events = unique_events.iloc[val_event_indices]

        # Create DataFrames for the current fold by merging back to df_raw
        df_train_fold = pd.merge(df_raw, train_events, on=grouping_keys, how='inner')
        df_val_fold = pd.merge(df_raw, val_events, on=grouping_keys, how='inner')

        if df_train_fold.empty or df_val_fold.empty:
            print(f"Warning: Fold {fold_idx+1} resulted in empty train or validation data. Skipping fold.")
            continue # Skip this fold

        # Create Dataset and DataLoader for the current fold
        # Root directories for processed data (optional, helps if pre_transform was used)
        train_root_fold = os.path.join('data', f'fold_{fold_idx+1}', 'train')
        val_root_fold = os.path.join('data', f'fold_{fold_idx+1}', 'val')


        
        # Fit scalers on the training data of this fold
        train_dataset_fold = ShipNoiseDataset(root=train_root_fold, df=df_train_fold,
                                            ship_features=SHIP_NODE_FEATURES,
                                            hydrophone_features=hydro_cols_for_dataset,
                                            target_col=TARGET_COLUMN,
                                            fit_scalers=True)
        
        # Use the scalers fitted on the training data for the validation data
        val_dataset_fold = ShipNoiseDataset(root=val_root_fold, df=df_val_fold,
                                          ship_features=SHIP_NODE_FEATURES,
                                          hydrophone_features=hydro_cols_for_dataset,
                                          target_col=TARGET_COLUMN,
                                          scalers=train_dataset_fold.scalers, # Pass the fitted scalers
                                          fit_scalers=False) # Do not refit
        
        df_train_fold.to_csv(train_root_fold+'\\train.csv', encoding='utf-8-sig')
        df_val_fold.to_csv(val_root_fold+'\\val.csv', encoding='utf-8-sig')

        train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        # Initialize model, optimizer, criterion, scheduler for THIS fold
        model = GNNNoisePredictor(
            num_hydrophone_features=sample_hydro_features_dim, # Use inferred dimension
            num_ship_features=sample_ship_features_dim,       # Use inferred dimension
            num_edge_features=sample_edge_features_dim,       # Use inferred dimension
            hidden_channels=HIDDEN_CHANNELS,
            num_gnn_layers=NUM_GNN_LAYERS
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.MSELoss()
        # Define scheduler lambda function again, it's scoped to this fold's model/optimizer
        def lr_lambda(epoch):
             # Adjust epoch indexing if scheduler steps per batch vs per epoch
             # Assuming step() is called after each epoch
            if epoch <= 3000:
                decay_factor = 0.1 + (0.9 * (3000 - epoch) / 3000)
                return decay_factor
            else:
                return 0.1 # Constant factor after 3000 epochs
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


        # Reset best train MSE tracker for this fold
        global BEST_TRAIN_MSE_FOR_FOLD
        BEST_TRAIN_MSE_FOR_FOLD = float('inf')
        
        print(f"Training model for Fold {fold_idx+1} for {NUM_EPOCHS_PER_FOLD} epochs...")
        
        # Training loop for the current fold
        # for epoch in tqdm(range(NUM_EPOCHS_PER_FOLD), desc=f"Fold {fold_idx+1} Training"):
        file = open(f'./data/fold_{fold_idx+1}/trainloss.txt', 'w', encoding='utf-8')
        for epoch in range(NUM_EPOCHS_PER_FOLD):

            # Pass the global variable reference or use global keyword inside train_fold if needed
            # Using global keyword inside the function is simpler here
            train_loss = train_fold(model, train_loader_fold, optimizer, scheduler, criterion, device, fold_idx, BEST_TRAIN_MSE_FOR_FOLD, epoch)
            file.write(f'{train_loss}\n')
            # The check and save logic are inside train_fold now
            
            # Optional: Evaluate on validation set periodically for monitoring
            if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS_PER_FOLD - 1: # Evaluate every 1000 epochs or at the end
                 val_avg_loss, val_mse, val_mae, val_r2, _, _ = evaluate(model, val_loader_fold, criterion, device)
                 print(f'\nFold {fold_idx+1}, Epoch {epoch+1:05d}: Train Loss: {train_loss:.4f} | '
                       f'Val Loss: {val_avg_loss:.4f}, Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}')

        file.close()
        # --- End of training for the current fold ---
        print(f"\n--- Finished Fold {fold_idx+1} ---")

        # Load the best model state dict saved for this fold
        model_save_path = os.path.join(SAVE_MODEL_DIR, f'best_train_mse_model_fold_{fold_idx+1}.pth')
        if os.path.exists(model_save_path):
             print(f"Loading best train MSE model for Fold {fold_idx+1} from {model_save_path}")
             model.load_state_dict(torch.load(model_save_path))
             model.to(device) # Ensure model is on the correct device after loading
        else:
             print(f"Warning: No best train MSE model saved for Fold {fold_idx+1}. Using the final model state.")


        # Evaluate the best model (or final model if no save happened) on the validation set of this fold
        print(f"Evaluating best model for Fold {fold_idx+1} on its validation set...")
        val_avg_loss, val_mse, val_mae, val_r2, y_val_actual, y_val_pred = evaluate(model, val_loader_fold, criterion, device)
        
        print(f"Fold {fold_idx+1} Validation Results: MSE={val_mse:.4f}, RMSE={np.sqrt(val_mse):.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'val_mse': val_mse,
            'val_rmse': np.sqrt(val_mse),
            'val_mae': val_mae,
            'val_r2': val_r2
        })

        all_test_actuals.extend(y_val_actual)
        all_test_preds.extend(y_val_pred)


    # --- End of K-Fold Loop ---

    print("\n--- K-Fold Cross-Validation Results ---")
    results_df = pd.DataFrame(fold_results)
    print(results_df)

    print("\n--- Average Results Across Folds ---")
    print(results_df[['val_mse', 'val_rmse', 'val_mae', 'val_r2']].mean())
    print("\n--- Standard Deviation Across Folds ---")
    print(results_df[['val_mse', 'val_rmse', 'val_mae', 'val_r2']].std())
    results_df.to_csv('result.csv', encoding='utf-8')
    # Plotting combined predictions from all folds' validation sets
    if all_test_actuals and all_test_preds:
        all_test_actuals = np.array(all_test_actuals)
        all_test_preds = np.array(all_test_preds)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(all_test_actuals, all_test_preds, alpha=0.3, label=f'Combined Val Data ({len(all_test_actuals)} points)')
        min_val = min(all_test_actuals.min(), all_test_preds.min()) if all_test_actuals.size > 0 and all_test_preds.size > 0 else 0
        max_val = max(all_test_actuals.max(), all_test_preds.max()) if all_test_actuals.size > 0 and all_test_preds.size > 0 else 1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        plt.xlabel("Actual " + TARGET_COLUMN); plt.ylabel("Predicted " + TARGET_COLUMN)
        plt.title(f"GNN K={K_FOLDS}-Fold CV: Actual vs. Predicted (Combined Validation Sets)"); plt.legend(); plt.grid(True)
        plt.savefig("gnn_kfold_actual_vs_predicted.png")
        print("\nCombined validation plot saved as gnn_kfold_actual_vs_predicted.png")
        plt.show()
    else:
        print("\nNo validation data processed across folds to generate scatter plot.")


if __name__ == '__main__':

    main_kfold_gnn()