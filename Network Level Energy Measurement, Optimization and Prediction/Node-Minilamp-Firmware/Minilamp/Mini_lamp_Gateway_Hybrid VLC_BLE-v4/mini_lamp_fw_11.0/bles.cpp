#include "delay.h"
#include "bles.h"
#include "serial_frame_util.h"
// extern const uint8_t ADDRESS_Source; // Source address

/**********Senhui Add**********/

uint8_t BLE_frameBuffer[BLE_FRAME_BUFFER_SIZE] = {0}; // Buffer to store received frame
uint8_t frame[BLE_FRAME_BUFFER_SIZE] = {0}; // Buffer to store sent frame
size_t frame_length = 0;                   // Length of the frame to send
size_t BLE_bufferHead = 0;  // Points to where new data is added
size_t BLE_bufferTail = 0;  // Points to where processing starts

volatile bool BLE_frameReceived = false;           // Flag indicating a complete frame received
// Define start and end frame markers
const uint8_t BLE_START_FRAME = 0xAA; // Start of Text (STX)
const uint8_t BLE_END_FRAME = 0xEE;   // End of Text (ETX)
// Addresses and commands for protocol
// const uint8_t BLE_ADDRESS_Source = {0xD6, 0xD6, 0x70, 0x4D, 0x29, 0x6B}; // ble address (Node).  
const uint8_t BLE_ADDRESS_Destination[6] = {0xD6, 0xD6, 0x70, 0x4D, 0x29, 0x6B}; // ble address (Node). 
// const uint8_t BLE_ADDRESS_Destination[6] = {0xFC, 0xE4, 0x82, 0xDC, 0xDD, 0x6D}; // ble address (Node). replace it with actual mac

const uint8_t CMD_SEND = 0x01;   // Example command to send data
const uint8_t BLE_CMD_RECEIVE = 0x02; // Example command to receive data
// Connection handle
uint16_t current_conn_handle = BLE_CONN_HANDLE_INVALID;
bool BLE_command_received = false;             // Command reception flag
bool BLE_response_frame_received = false;             // frame reception flag
bool isConnected = false; // Flag to track connection status


// <<< ADDED >>> Global variables for connection state and parameters
uint16_t _conn_handle = BLE_CONN_HANDLE_INVALID; // Store the connection handle
// Store the *last requested* values to avoid redundant requests
// Initialize with values that likely won't match the first request
int last_requested_phy_pref = 0;       // 0: Unset, 1: 1M, 2: 2M, 3: Coded
uint16_t last_requested_interval_units = 0; // In 1.25ms units
uint16_t last_requested_mtu = 0;       // MTU size (default is usually 23)


const uint8_t CMD_Running_State = 0x03; //Power mode
const uint8_t CMD_Wakeup = 0x04; //Wakeup 
const uint8_t CMD_Communcation_mode = 0x05; //Communcation mode, choose vlc or ble
const uint8_t CMD_VLC_Protocol = 0x06; //vlc bit, byte, compact level
const uint8_t CMD_VLC_Intervel = 0x07; //Intervel
const uint8_t CMD_PWM = 0x08; //PWM
const uint8_t CMD_VLC_Unit = 0x09; //MTU
const uint8_t CMD_BLE_Protocol = 0x0A; //BLE_Protocol
const uint8_t CMD_BLE_Intervel = 0x0B; //BLE_Intervel
const uint8_t CMD_PHY_Rate = 0x0C; //PHY_Rate
const uint8_t CMD_MTU = 0x0D; //MTU
const uint8_t CMD_BLE_Unit = 0x0E; //Unit
const uint8_t CMD_BLE_OFF_ON = 0x0F; //BLE OFF/ON
const uint8_t CMD_BLE_FAST_TIME = 0X11;

const uint8_t CMD_VLC_OFF_ON = 0x0A; //VLC OFF/ON
const uint8_t CMD_EINK_OFF_ON = 0x0B; //VLC OFF/ON
const uint8_t CMD_SENSING_OFF_ON = 0x0C; //VLC OFF/ON

const uint8_t CMD_VLC_SEND_Auto = 0x10; //SEND VLC AUTO FOR TEST
const uint8_t CMD_BLE_SEND_Auto = 0x20; //SEND BLE AUTO FOR TEST

// State machine states
enum FrameState {
    STATE_WAIT_HEADER,
    STATE_RECEIVE_ADDRESS,
    STATE_RECEIVE_CMD,
    STATE_RECEIVE_LENGTH,
    STATE_RECEIVE_DATA,
    STATE_RECEIVE_CHECKSUM,
    STATE_WAIT_END,
    STATE_FRAME_COMPLETE
};
#define ADDRESS_SIZE 6
#define MAX_DATA_SIZE 255
FrameState frameState = STATE_WAIT_HEADER;
uint16_t BLE_frameIndex = 0;              // Current index in the frame buffer
uint16_t frameDataLength = 0; // The expected data length
uint8_t frameChecksum = 0; // Calculated checksum
bool isFrameComplete = false; // Flag to indicate frame completion
uint8_t frame_calculatedChecksum = 0;

unsigned long last_ble_activity_time = 0;
const unsigned long BLE_IDLE_TIMEOUT = 10000; // 10 seconds

extern bool record_time_flag;
void response_callback(BLEClientCharacteristic* chr, uint8_t* data, uint16_t len);
void processFramePayload(size_t start, size_t frameSize);
void send_respond_BLEFrame(uint8_t* addr);

void processFrame(uint8_t* frame, uint16_t length);
uint8_t BLE_calculateChecksum(uint8_t* data, size_t length);
/**********Senhui Add**********/


int update_char = 0;
int scan_type = SCAN_TYPE_NULL;
// uint8_t target_mac[6] = {0x49, 0xAD, 0x2D, 0xE2, 0x33, 0xF7};  // MAC address in little endian
uint8_t target_mac[6] = {0xD6, 0xD6, 0x70, 0x4D, 0x29, 0x6B};  // MAC address in little endian
// Define the 8-bit value you want to write
uint8_t value_to_write = 0x01;  // Replace 0x01 with the value you want to write

BLEClientService        hrms("19b10010e8f2537e4f6cd104768a1214");
BLEClientCharacteristic hrmc("19b10020e8f2537e4f6cd104768a1214");

/************Senhui Add************/
bool command_received = false;     // Command reception flag
BLEClientCharacteristic charRX("19b10020e8f2537e4f6cd104768a12ee"); // For sending
BLEClientCharacteristic charTX("19b10020e8f2537e4f6cd104768a12aa"); // For receiving
/************Senhui Add************/

void bles_setup() 
{
  // Initialize Bluefruit with maximum connections as Peripheral = 0, Central = 1
  // SRAM usage required by SoftDevice will increase dramatically with number of connections
  Bluefruit.begin(0, 1);
  Bluefruit.setTxPower(0);    // Check bluefruit.h for supported values

  collected_data.spare_4 |= (1 << 2); //bit 2 becomes 1  TX=-20dBm, -16dBm, -12dBm, -8dBm, -4dBm, 0dBm, +3dBm and +4dBm.
  collected_data.spare_4 |= (1 << 3); //bit 3 becomes 1, 0b00011100
  collected_data.spare_4 |= (1 << 4); //bit 4 becomes 1, 0b00011100

  /************Senhui Add************/
  // Format the node name with the source address
  // Extract the lower nibble
  int ADDRESS_Source =2;
  uint8_t lowerNibble = ADDRESS_Source & 0x0F;
  // Format the node name with the lower nibble
  char nodeName[20]; // Buffer to hold the formatted string
  snprintf(nodeName, sizeof(nodeName), "MiniGW #%02d", lowerNibble);
  // Set the name
  Bluefruit.setName(nodeName);
  /************Senhui Add************/

  // Bluefruit.setName("MiniGW");
  // Initialize HRM client
  hrms.begin();
  // hrmc.begin();

  /************Senhui Add************/
  charRX.begin();
  charTX.begin();
  // Start Central Scan
  Bluefruit.setConnLedInterval(250);
  // Callbacks for Central
  Bluefruit.Scanner.setRxCallback(scan_callback);
  Bluefruit.Central.setDisconnectCallback(disconnect_callback);
  Bluefruit.Central.setConnectCallback(connect_callback);
// Bluefruit.Scanner.start(0);

  Bluefruit.Scanner.restartOnDisconnect(true);
  Bluefruit.Scanner.setInterval(100, 100); // in unit of 0.625 ms.  Every 100 ms (160×0.625 ms=100 ms), wake up and listen for 50 ms (80×0.625 ms=50 ms) for any BLE advertisements, then sleep until the next 100 ms boundary.
  Bluefruit.Scanner.filterUuid(hrms.uuid);
  Bluefruit.Scanner.useActiveScan(false);
}

void bles_start_scan(int sec, int type, String * mac, uint8_t data)
{
    scan_type = type;
    if( type == SCAN_TYPE_WRITE )
    {

      int i;
        // Convert the mac string to the target_mac array
      int index = 0;
      Serial.println(*mac);
      char *token = strtok((char*)mac->c_str(), ":");

      while (token != nullptr && index < 6) {
        target_mac[index++] = strtol(token, nullptr, 16);
        token = strtok(nullptr, ":");
      }
        
      value_to_write = data;    
      update_char = 1;
    }
    Bluefruit.Scanner.start(sec);
}


int bles_written()
{
  if( !update_char ) return 1; else return 0;
}

void bles_stop_scan()
{
  scan_type = SCAN_TYPE_NULL;
  Bluefruit.Scanner.stop();
}

bool bles_is_on()
{
    return Bluefruit.Scanner.isRunning();
}

void scan_callback(ble_gap_evt_adv_report_t* report)
{
  // MAC is in little endian --> print reverse
  if(scan_type == SCAN_TYPE_LIST) 
  {
    Serial.printBufferReverse(report->peer_addr.addr, 6, ':');
    Serial.print(",");

    Serial.print(report->rssi);
    Serial.print(",");

    Serial.printBuffer(report->data.p_data, report->data.len, '-');
    /*
    Serial.print(",T=");
    float T = (report->data.p_data[8] + report->data.p_data[9]*256.0)*.005;
    float RH = (report->data.p_data[10] + report->data.p_data[11]*256.0)*.0025;
    Serial.print(T);   Serial.print(",RH%="); Serial.print(RH);*/
    Serial.println();
    Bluefruit.Scanner.resume();
  } else
    if( scan_type == SCAN_TYPE_WRITE)
    {
      if( !update_char )
      {
        Bluefruit.Scanner.resume();
        return;
      }
      // Check if the MAC address matches the target MAC address
      bool mac_match = true;
      for (int i = 0; i < 6; i++) {
        if (report->peer_addr.addr[i] != target_mac[5-i]) {
          mac_match = false;
          break;
        }
      }

      // Connect only if MAC address matches
      if (mac_match) {
        update_char = 0; //only update characteristic once
        Bluefruit.Central.connect(report);
      } else {
        Bluefruit.Scanner.resume();
      }      
    }
  // For Softdevice v6: after received a report, scanner will be paused
  // We need to call Scanner resume() to continue scanning
  Bluefruit.Scanner.resume();
}

/**
 * Callback invoked when an connection is established
 * @param conn_handle
 */
void connect_callback(uint16_t conn_handle)
{
  isConnected = true; // Set the flag to true on successful connection
  record_time_flag = false;
  current_conn_handle = conn_handle; // Save the connection handle
  // If HRM is not found, disconnect and return
  if ( !hrms.discover(conn_handle) )
  {
    // disconnect since we couldn't find HRM service
    Bluefruit.disconnect(conn_handle);
    return;
  }


  // // Once HRM service is found, we continue to discover its characteristic  
  // if ( !hrmc.discover() )
  // {
  //   // Measurement characteristic is mandatory; if it is not found (valid), then disconnect
  //   Bluefruit.disconnect(conn_handle);
  //   return;
  // }

  // // Write the 8-bit value to the characteristic
  // if (hrmc.write8(value_to_write)) // write twice to be sure
  // {
  //   //Serial.println("8-bit value written successfully.");
  // } 
  // else 
  // {
  //   //Serial.println("Failed to write 8-bit value to characteristic.");
  // }

  Serial.println("Successfully connected to the node!");

    if (!charRX.discover() || !charTX.discover()) {
    Serial.println("Characteristic discovery failed. Disconnecting...");
    Bluefruit.disconnect(conn_handle);
    return;
  }

  // Send the frame via charRX
  if (charRX.write(frame, frame_length)) {
    last_ble_activity_time = millis(); // Reset timer after sending
    collected_data.ble_communication_volume = collected_data.ble_communication_volume + frame_length;
    collected_data.ble_information_volume = collected_data.ble_information_volume + frame_length - 11;
    //set 0 bit, reprsent sending
    collected_data.running_mode |= (1 << 0); 
    send_periodic_frame_example();
    Serial.println("Frame sent successfully.");
  } else {
    Serial.println("Failed to send frame. Disconnecting...");
    Bluefruit.disconnect(conn_handle);
    return;
  }

  // Set the notification callback and enable notifications to receive a response
  charTX.setNotifyCallback(response_callback);
  if (!charTX.enableNotify()) {
    Serial.println("Failed to enable notifications. Disconnecting...");
    Bluefruit.disconnect(conn_handle);
  }
  delay(1000);
  // Bluefruit.disconnect(conn_handle);
}

volatile bool ble_frameFlag = false;    // Flag indicating a complete ble frame received
// Callback to process received data. Since ble only can receive about 20 bytes, it will run this function several times when it receive the long frame. 
// Therefore, state machine method can solve the long frame. Don't put other function into response_callback, it will not receive long frame!
void response_callback(BLEClientCharacteristic* chr, uint8_t* data, uint16_t len) {
    last_ble_activity_time = millis(); // Reset timer after receiving
    collected_data.spare_2 = collected_data.spare_2 + len;
    // CLear bit 1 . maybe impact ble to receive the data?
    collected_data.running_mode &= ~(1 << 6);
    send_periodic_frame_example(); //send node status
    for (uint16_t i = 0; i < len; i++) {
        uint8_t byte = data[i];
        Serial.print(data[i], HEX);
        Serial.print(" ");
        switch (frameState) {
            case STATE_WAIT_HEADER:
                if (byte == BLE_START_FRAME) {
                    BLE_frameIndex = 0;
                    frameChecksum = 0;
                    BLE_frameBuffer[BLE_frameIndex++] = byte;
                    frameState = STATE_RECEIVE_ADDRESS;
                }
                break;

            case STATE_RECEIVE_ADDRESS:
                BLE_frameBuffer[BLE_frameIndex++] = byte;
                if (BLE_frameIndex == ADDRESS_SIZE+1) {
                    frameState = STATE_RECEIVE_CMD;
                }
                break;

            case STATE_RECEIVE_CMD:
                BLE_frameBuffer[BLE_frameIndex++] = byte;
                frameState = STATE_RECEIVE_LENGTH;
                break;

            case STATE_RECEIVE_LENGTH:
                BLE_frameBuffer[BLE_frameIndex++] = byte;
                frameDataLength = byte; // Length of data
                if (frameDataLength > MAX_DATA_SIZE - (ADDRESS_SIZE + 4)) { // Validate length
                    Serial.println("Invalid frame length, resetting...");
                    frameState = STATE_WAIT_HEADER;
                } else {
                    frameState = STATE_RECEIVE_DATA;
                }
                break;

            case STATE_RECEIVE_DATA:
                BLE_frameBuffer[BLE_frameIndex++] = byte;
                if (BLE_frameIndex == ADDRESS_SIZE + 3 + frameDataLength) { // Header + Address (6) + cmd (1) + length (1) + data
                    frameState = STATE_RECEIVE_CHECKSUM;
                }
                break;

            case STATE_RECEIVE_CHECKSUM:
                BLE_frameBuffer[BLE_frameIndex++] = byte;
                frameChecksum = byte;
                // Validate checksum
                // frame_calculatedChecksum = BLE_calculateChecksum(BLE_frameBuffer, BLE_frameIndex - 1); // Exclude received checksum
                for (size_t i = 0; i < (BLE_frameIndex-2); i++) {
                    frame_calculatedChecksum ^= BLE_frameBuffer[i];
                }
                // Serial.print("frame_calculatedChecksum");
                // Serial.print(frame_calculatedChecksum, HEX);
                // Serial.print(" ");
                // Serial.print("frameChecksum");
                // Serial.print(frameChecksum, HEX);
                // Serial.print(" ");
                if (frame_calculatedChecksum == frameChecksum) {
                    frameState = STATE_WAIT_END;
                } else {
                    Serial.println("Checksum mismatch, resetting...");
                    frameState = STATE_WAIT_HEADER;
                    BLE_command_received = false;  // disconnect the node
                }
                frameState = STATE_WAIT_END;  
                break;

            case STATE_WAIT_END:
                BLE_frameBuffer[BLE_frameIndex++] = byte;
                if (byte == BLE_END_FRAME) {
                    isFrameComplete = true;
                    frameState = STATE_FRAME_COMPLETE;
                } else {
                    Serial.println("Invalid frame end, resetting...");
                    frameState = STATE_WAIT_HEADER;
                    BLE_command_received = false;  // disconnect the node
                }
                break;

            case STATE_FRAME_COMPLETE:
                Serial.println();
                Serial.println("Received frame successfully!");
                processFrame(BLE_frameBuffer, BLE_frameIndex); // Process the complete frame
                ble_frameFlag = true;
                frameState = STATE_WAIT_HEADER; // Reset state machine
                break;

            default:
                frameState = STATE_WAIT_HEADER;
                break;
        }
    }
}

/**
 * Callback invoked when a connection is dropped
 * @param conn_handle
 * @param reason is a BLE_HCI_STATUS_CODE which can be found in ble_hci.h
 */
void disconnect_callback(uint16_t conn_handle, uint8_t reason)
{
  isConnected = false; // Reset the flag on disconnection
  record_time_flag = false;
  Serial.println("Disconnected from node.");
  current_conn_handle = BLE_CONN_HANDLE_INVALID; // Reset connection handle
  (void) conn_handle;
  (void) reason;

  //Serial.print("Disconnected, reason = 0x"); Serial.println(reason, HEX);
}



//######################senhui added###################
void CMD_BLE_Intervel_Process()
{
  Bluefruit.Scanner.setInterval(uint16_t(collected_data.ble_interval_seconds), uint16_t(collected_data.ble_interval_seconds*0.5));
}

void CMD_PHY_Rate_Process()
{
  int desired_phy_pref;         // 1 for 1M, 2 for 2M, 3 for Coded
  if (collected_data.phy_rate_percentage == PHY_1M) { // High value on param1 -> High throughput
    desired_phy_pref = PHY_1M;
  }else if (collected_data.phy_rate_percentage == PHY_2M) {
    desired_phy_pref = PHY_2M; // Or PHY_CODED if you need long range
  }else if (collected_data.phy_rate_percentage == PHY_3M) {
    desired_phy_pref = PHY_3M; // Or PHY_CODED if you need long range
  }
    else { 
    desired_phy_pref = PHY_1M; 
    collected_data.phy_rate_percentage = PHY_1M;
  }

  // ---  Parameter Update Logic ---
  if (_conn_handle != BLE_CONN_HANDLE_INVALID) {
    // We are connected
    BLEConnection* connection = Bluefruit.Connection(_conn_handle);

    // Ensure connection object is still valid
    if (connection && connection->connected()) {

      bool param_update_requested = false;

      // --- Check and Request PHY Update ---
      if (desired_phy_pref != last_requested_phy_pref) {
        Serial.print("PHY preference changed. Requesting: ");
        uint8_t req_phy_mask = 0;
        switch(desired_phy_pref) {
            case PHY_1M:    Serial.println("1M"); req_phy_mask = BLE_GAP_PHY_1MBPS; break;
            case PHY_2M:    Serial.println("2M"); req_phy_mask = BLE_GAP_PHY_2MBPS; break;
            case PHY_3M: Serial.println("Coded (Long Range)"); req_phy_mask = BLE_GAP_PHY_CODED; break;
            default:        Serial.println("Default/Auto"); req_phy_mask = BLE_GAP_PHY_AUTO; break; // Should not happen with good logic
        }

        // Request symmetric PHY (same for TX and RX)
        // You can specify different PHYs for tx & rx if needed: requestPHY(tx_phy, rx_phy, options)
        if (connection->requestPHY(req_phy_mask)) {
            last_requested_phy_pref = desired_phy_pref; // Update last requested value *only if request was sent*
            param_update_requested = true;
        } else {
            Serial.println("Failed to *send* PHY change request.");
        }
      }

      else if (connection && !connection->connected()) {
      // Connection object exists but is no longer connected (should be handled by disconnect_callback, but good safety check)
      _conn_handle = BLE_CONN_HANDLE_INVALID;
      }
    }
  }
}

void CMD_MTU_Process()
{
  uint16_t desired_mtu;         // Desired MTU
  if (_conn_handle != BLE_CONN_HANDLE_INVALID) {
    BLEConnection* connection = Bluefruit.Connection(_conn_handle);

    if (connection && connection->connected()) {
      bool update_requested_this_loop = false; 
      
      desired_mtu = max((uint16_t)23, min((uint16_t)247, collected_data.mtu_value)); // Clamp MTU

      collected_data.mtu_value = desired_mtu;

      if (desired_mtu != last_requested_mtu) {
      Serial.print("MTU Preference Changed. Requesting: "); Serial.println(desired_mtu);
      Serial.println("--> Also requesting Data Length Update (DLE).");
      connection->requestDataLengthUpdate(); // Good practice with MTU change

      if (connection->requestMtuExchange(desired_mtu)) {
        last_requested_mtu = desired_mtu;
        update_requested_this_loop = true;
      } else {
        Serial.println("--> Failed to send MTU change request.");
      }
    }
  }else if (connection && !connection->connected()) {
    // Connection object exists but is no longer connected (should be handled by disconnect_callback, but good safety check)
    _conn_handle = BLE_CONN_HANDLE_INVALID;
  }

  }

}


unsigned long lastResetTime = millis();
// Function to report bytes sent in the last 10 seconds and reset the counter
void reportAndResetBytesCount() {
  // Serial.print("Bytes sent in the last unit seconds: ");
  // Serial.println(collected_data.ble_communication_volume);
  
  // Reset the counter
  collected_data.ble_communication_volume = 0;
  collected_data.ble_information_volume = 0;
  lastResetTime = millis();
  send_periodic_frame_example(); //send node status
}

unsigned long lastResetTime_RX = millis();
// Function to report bytes sent in the last 10 seconds and reset the counter
void reportAndResetBytesCount_RX() {
  // Serial.print("Bytes sent in the last unit seconds: ");
  // Serial.println(collected_data.ble_communication_volume);
  
  // Reset the RX counter
  collected_data.spare_2 = 0;
  lastResetTime_RX = millis();
  send_periodic_frame_example(); //send node status
}









uint8_t BLE_calculateChecksum(uint8_t* data, size_t length) {
    uint8_t checksum = 0;
    for (size_t i = 0; i < length; i++) {
        checksum ^= data[i];
    }
    return checksum;
}



void sendBLEFrame(uint8_t* data, int count) {
    size_t index = 0;
    uint8_t start_index = 5;
    // Start of frame
    frame[index++] = BLE_START_FRAME;
    // Address (2 bytes)
    // frame[index++] = BLE_ADDRESS_Destination;
    frame[index++] = data[0]; // ADDRESS_Destination 0
    frame[index++] = data[1]; // ADDRESS_Destination 1 
    frame[index++] = data[2]; // ADDRESS_Destination 2
    frame[index++] = data[3]; // ADDRESS_Destination 3
    frame[index++] = data[4]; // ADDRESS_Destination 4
    frame[index++] = data[5]; // ADDRESS_Destination 5

    for (uint8_t i=0; i<6; i++)
    {
          target_mac[i] = data[i];
          // Serial.print("0x");
          // Serial.print(target_mac[i], HEX);
          // Serial.print(" ");
    }
    // Command
    // frame[index++] = CMD_SEND;
    frame[index++] = data[start_index+1]; 
    if (data[start_index+1] == BLE_CMD_RECEIVE)
    {
      count = 0;// only require node send data to gateway
    }
    // Number of data bytes
    frame[index++] = count * sizeof(uint8_t);
    // Copy actual data (just copy the uint8_t data as is)
    for (int i = 0; i < count; i++) {
        frame[index++] = data[start_index+2+i];  // Copy each uint8_t byte
    }
    // Checksum
    frame[index++] = BLE_calculateChecksum(&frame[0], index - 1);
    // End of frame
    frame[index++] = BLE_END_FRAME;
    frame_length = index;
    // Send the frame using send_NEC
    Serial.print("Send BLE frame: ");
    for (size_t i = 0; i < index; i++) {
      // Print the frame byte in hexadecimal
      Serial.print(frame[i], HEX);
      Serial.print(" ");
    }
    Serial.println(" ");
    // Serial.println("Sent successfully");
}


void send_respond_BLEFrame(uint8_t* addr) {
    size_t index = 0;
    uint8_t start_index = 5;
    uint8_t count = 0;
    uint8_t data[6] = {0};
    // Start of frame
    frame[index++] = BLE_START_FRAME;
    // Address (2 bytes)
    // frame[index++] = BLE_ADDRESS_Destination;
    frame[index++] = addr[0]; // ADDRESS_Destination 0
    frame[index++] = addr[1]; // ADDRESS_Destination 1 
    frame[index++] = addr[2]; // ADDRESS_Destination 2
    frame[index++] = addr[3]; // ADDRESS_Destination 3
    frame[index++] = addr[4]; // ADDRESS_Destination 4
    frame[index++] = addr[5]; // ADDRESS_Destination 5
    // Command
    frame[index++] = CMD_SEND;
    // Number of data bytes
    frame[index++] = count * sizeof(uint8_t);
    // Copy actual data (just copy the uint8_t data as is)
    for (int i = 0; i < count; i++) {
        frame[index++] = data[start_index+1+i];  // Copy each uint8_t byte
    }
    // Checksum
    frame[index++] = BLE_calculateChecksum(&frame[0], index - 1);
    // End of frame
    frame[index++] = BLE_END_FRAME;
    frame_length = index;
    // Send the frame using send_NEC
    Serial.print("Response BLE frame: ");
    for (size_t i = 0; i < index; i++) {
      // Print the frame byte in hexadecimal
      Serial.print(frame[i], HEX);
      Serial.print(" ");
    }
    Serial.println(" ");
    Serial.println("Sent response BLE frame successfully");
}

// Function to extract data from the command string
bool extractDataFromCmd(const String& cmd, uint8_t** send_data, int& data_len) {
  // Find the position of the first comma
  int commaPos = cmd.indexOf(',');
  if (commaPos == -1) {
    Serial.println("Invalid command format: No comma found.");
    return false;
  }
  // Extract the data part after the comma
  String dataPart = cmd.substring(commaPos + 1);
  // Count the number of data fields using a traditional for loop
  int count = 1; // Start with 1 because there's at least one field
  for (int i = 0; i < dataPart.length(); i++) {
    if (dataPart.charAt(i) == ',') {
      count++;
    }
  }
  // Allocate memory dynamically based on the number of fields
  *send_data = (uint8_t*)malloc(count * sizeof(uint8_t));
  if (*send_data == nullptr) {
    Serial.println("Memory allocation failed.");
    return false;
  }
  // Parse the data fields
  int index = 0;
  char* token = strtok((char*)dataPart.c_str(), ",");
  while (token != nullptr) {
    (*send_data)[index++] = (uint8_t)strtol(token, nullptr, 16); // Convert to uint8_t
    token = strtok(nullptr, ",");
  }
  // Update the length of the extracted data
  data_len = index;
  return true;
}


// BLE_Data processFramePayload();


BLE_Data processFramePayload() {
    uint8_t addr_ble[6]={0};
    size_t index = 0;  
    addr_ble[0] = BLE_frameBuffer[index];
    addr_ble[1] = BLE_frameBuffer[(index + 1) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[2] = BLE_frameBuffer[(index + 2) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[3] = BLE_frameBuffer[(index + 3) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[4] = BLE_frameBuffer[(index + 4) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[5] = BLE_frameBuffer[(index + 5) % BLE_FRAME_BUFFER_SIZE];

    uint8_t command = BLE_frameBuffer[(index + 6) % BLE_FRAME_BUFFER_SIZE];
    uint8_t length = BLE_frameBuffer[(index + 7) % BLE_FRAME_BUFFER_SIZE];

    BLE_Data data = {{addr_ble[0], addr_ble[1], addr_ble[2], addr_ble[3], addr_ble[4], addr_ble[5]}, command, length, index};
    return data;
}


void processFrame(uint8_t* frame, uint16_t length) {
    uint8_t addr_ble[6]={0};
    // Serial.println("Valid frame received!");

    // Extract address
    Serial.print("Address: ");
    for (uint8_t i = 1; i <= 6; i++) {
        addr_ble[i]=frame[i];
        Serial.print(frame[i], HEX);
        if (i < 6) Serial.print(":");
    }
    Serial.println();

    // Extract command
    uint8_t command = frame[7];
    Serial.print("Command: 0x");
    Serial.println(command, HEX);

    // Extract data
    uint8_t dataLength = frame[8];
    Serial.print("Data: ");
    for (uint8_t i = 0; i < dataLength; i++) {
        Serial.print(frame[9 + i], HEX);
        Serial.print(" ");
    }
    Serial.println();

    if (command == CMD_SEND)
    {
      // Print or handle the received frame
      // Serial.print("No respond for this frame!");
      // Serial.println();
      ;

    }
    else if (command == BLE_CMD_RECEIVE)//send data to node
    {
      send_respond_BLEFrame(addr_ble);
      delay(1000);
    }
    BLE_command_received = false;  // disconnect the node
}








