"""
Research Data Upload Module - ThingSpeak Bulk Upload
Function: Upload data from data_log.txt to ThingSpeak platform
"""

from machine import UART, Pin
import utime
from math import ceil
from binascii import hexlify
import os

# ==================== Hardware Configuration ====================
# Network control pin
NET_PIN = 3
# UART configuration (for 4G module communication)
uart1 = UART(1, 115200, bits=8, tx=Pin(4), rx=Pin(5), txbuf=4096, rxbuf=4096)


# ==================== Helper Functions ====================
def str_to_hexStr(string):
    """Convert string to hexadecimal string"""
    str_bin = string.encode('utf-8')
    return hexlify(str_bin).decode('utf-8')


def count_lines(filename='data_log.txt'):
    """Count the number of lines in a file"""
    try:
        os.stat(filename)
    except:
        return 0
    try:
        with open(filename, 'r') as file:
            line_count = 0
            while file.readline():
                line_count += 1
            return line_count
    except OSError:
        return -1


def analyze_lines(filename):
    """Analyze file, get line count and character length"""
    try:
        with open(filename, 'r') as file:
            line_count = 0
            line_lengths = 0
            for line in file:
                length = len(line.strip())
                line_lengths += length
                line_count += 1
                if line_count >= 200:
                    break
            return {
                'total_lines': line_count,
                'lengths': line_lengths
            }
    except OSError:
        return None


def read_lines_range(filename, start_line, end_line):
    """Read a specific range of lines from a file"""
    try:
        with open(filename, 'r') as file:
            if start_line < 1 or end_line < start_line:
                return []
            lines = []
            current_line = 1
            while current_line < start_line:
                if not file.readline():
                    return []
                current_line += 1
            while current_line <= end_line:
                line = file.readline()
                if not line:
                    break
                lines.append(line.strip())
                current_line += 1
            return lines
    except OSError:
        print(f"Cannot open file: {filename}")
        return []


def sendCMD_waitResp(cmd, timeout=3000):
    """Send AT command and wait for response"""
    start_time = utime.ticks_ms()
    uart1.write((cmd + '\r\n').encode())
    return waitResp(timeout, start_time)


def waitResp(timeout, start_time):
    """Wait for response data"""
    resp = b""
    while (utime.ticks_ms() - start_time) < timeout:
        if uart1.any():
            resp = b"".join([resp, uart1.read(1)])
    try:
        decoded_resp = resp.decode()
        print(decoded_resp)
        return decoded_resp
    except UnicodeError:
        pass
    return ""


def netOn(on=True):
    """Control network module power"""
    pwr_key = Pin(NET_PIN, Pin.OUT)
    
    if 'OK' in sendCMD_waitResp('AT', timeout=400):
        if on == True:
            return
        pwr_key.value(1)
        utime.sleep(5)
        pwr_key.value(0)
        print('Network module off')
        return
    elif on == False:
        return
    
    pwr_key.value(1)
    utime.sleep(4)
    pwr_key.value(0)
    utime.sleep(25)


# ==================== Core Upload Function ====================
def process_data_log(filename='data_log.txt', 
                     channel_id='YOUR_CHANNEL_ID', 
                     write_api_key='YOUR_WRITE_API_KEY'):
    """
    Upload data file to ThingSpeak platform in bulk
    
    Parameters:
        filename: Data file name, default 'data_log.txt'
        channel_id: ThingSpeak Channel ID (get from thingspeak.com)
        write_api_key: ThingSpeak Write API Key (get from thingspeak.com)
    
    Returns:
        bool: True if upload successful, False otherwise
    """
    total_lines = count_lines(filename)
    
    # Control upload frequency: upload every 100 data records
    if total_lines >= 100 and ((total_lines % 100 == 0) or 
                                (total_lines % 100 == 1) or 
                                (total_lines % 100 == 2) or 
                                (total_lines % 100 == 5)):
        None
    else:
        return False

    # Power on network module
    netOn(True)
    
    # Analyze file data
    result = analyze_lines(filename)
    if not result:
        print('File analysis failed')
        return False
        
    length = result['lengths']
    rows = min(result['total_lines'], 200)  # Upload max 200 rows per batch
    
    # Build upload data (ThingSpeak bulk update format)
    # Format: write_api_key=xxx&time_format=absolute&updates=data1|data2|data3
    unhex = f'write_api_key={write_api_key}&time_format=absolute&updates=' + \
            '|'.join(read_lines_range(filename, 1, 3))
    hexs = str_to_hexStr(unhex)
    dataLength = 2 * (length + rows - 1) + 120
    totalLength = dataLength + 85 + len(str(dataLength)) + 1
    
    # Create HTTP connection
    if 'ERROR' in sendCMD_waitResp(f'AT+CHTTPCREATE="http://api.thingspeak.com"', timeout=3000):
        return False
    
    # Establish connection
    if 'ERROR' in sendCMD_waitResp("AT+CHTTPCON=0", timeout=3000):
        sendCMD_waitResp("AT+CHTTPDESTROY=0", timeout=600)
        return False
    
    # Send HTTP request header
    send_time = utime.ticks_ms()
    sendCMD_waitResp(f"AT+CHTTPSENDEXT=1,{totalLength},85,0,1,33,\"/channels/{channel_id}/bulk_update.csv\",0,,33,\"application/x-www-form-urlencoded\",", timeout=800)
    
    # Send data in chunks
    suc = False
    back = ''
    for i in range(ceil(rows / 3)):
        index = int(i != ceil(rows/3) - 1)
        if i == 0:
            back = sendCMD_waitResp(f"AT+CHTTPSENDEXT={index},{totalLength},{len(hexs) + 1 + len(str(dataLength))},{dataLength},{hexs}", 
                                    timeout=500+(1-index)*16000)
        else:
            unhex = '|' + '|'.join(read_lines_range(filename, i * 3 + 1, min(i * 3 + 3, rows)))
            hexs = str_to_hexStr(unhex)
            back = sendCMD_waitResp(f"AT+CHTTPSENDEXT={index},{totalLength},{len(hexs)},{hexs}", 
                                    timeout=500+(1-index)*40000)
    
    # Disconnect
    sendCMD_waitResp("AT+CHTTPDISCON=0", timeout=600)
    sendCMD_waitResp("AT+CHTTPDESTROY=0", timeout=600)
    netOn(False)
    
    # Process upload result
    if "202 Accepted" in back:
        try:
            if rows >= total_lines:  # If all lines were sent, delete the file
                os.remove(filename)
            else:  # Otherwise remove only the sent lines
                with open(filename, 'r') as src, open("temp.txt", 'w') as tmp:
                    current_line = 0
                    for line in src:
                        current_line += 1
                        if current_line > rows:
                            tmp.write(line)
                os.remove(filename)
                os.rename("temp.txt", filename)
            suc = True
            print("Data upload successful!")
        except OSError:
            print("Error processing file")
    
    return suc


# ==================== Usage Example ====================
def example_upload():
    """
    Usage example
    """
    # Configure your ThingSpeak credentials (DO NOT commit these to GitHub!)
    # Get your credentials from: https://thingspeak.com/channels
    CHANNEL_ID = 'YOUR_CHANNEL_ID'           # Replace with your Channel ID
    WRITE_API_KEY = 'YOUR_WRITE_API_KEY'     # Replace with your Write API Key
    
    # Execute upload
    success = process_data_log(
        filename='data_log.txt',
        channel_id=CHANNEL_ID,
        write_api_key=WRITE_API_KEY
    )
    
    if success:
        print("Data successfully uploaded to ThingSpeak")
    else:
        print("Upload failed or conditions not met")


# Run test
if __name__ == '__main__':
    example_upload()
