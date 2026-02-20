#include "serial_frame_util.h"
#include <stdio.h> // For checksum calculation (though not strictly needed for simple sum)

// Define header and end bytes (you can also define these in the header if you want)
#define HEADER_BYTE 0xAA
#define END_BYTE 0xEE



// void send_periodic_frame_example() {
//     static unsigned long lastSendTime = 0;
//     unsigned long currentTime = millis();
//     int time_seconds = 0;

//     if (currentTime - lastSendTime >= 1000) { // Send every 1 seconds
//         lastSendTime = currentTime;
//         time_seconds = int(lastSendTime / 1000);

         
//         // Example data bytes to send (replace with your actual data - NOW HEX BYTES)
//         byte dataToSend[20]; // Declare as array of 14 bytes
//         dataToSend[0] = collected_data.power_mode;
//         dataToSend[1] = collected_data.wakeup_state;
//         dataToSend[2] = collected_data.communication_mode;
//         dataToSend[3] = collected_data.temperature;
//         dataToSend[4] = collected_data.vlc_protocol_mode;
//         dataToSend[5] = collected_data.vlc_interval_seconds;
//         dataToSend[6] = collected_data.vlc_communication_volume;
//         dataToSend[7] = collected_data.vlc_information_volume;
//         dataToSend[8] = collected_data.vlc_per_unit_time;
//         dataToSend[9] = collected_data.pwm_percentage; 
//         dataToSend[10] = collected_data.ble_protocol_mode; 
//         dataToSend[11] = collected_data.ble_interval_seconds; 
//         dataToSend[12] = collected_data.ble_communication_volume; 
//         dataToSend[13] = collected_data.ble_information_volume; 
//         dataToSend[14] = collected_data.ble_per_unit_time; 
//         dataToSend[15] = collected_data.phy_rate_percentage; 
//         dataToSend[16] = collected_data.mtu_value;  
//         dataToSend[17] = collected_data.spare_2; 
//         dataToSend[18] = collected_data.spare_3; 
//         dataToSend[19] = collected_data.spare_4;  
//         dataToSend[20] = (byte)time_seconds; 

//         size_t dataLength = sizeof(dataToSend) / sizeof(dataToSend[0]);

//         // // --- DEBUG PRINTS ---
//         // Serial.println("\n--- Sending Frame ---");
//         // Serial.print("Data to Send (dataToSend): ");
//         for (size_t i = 0; i < dataLength; i++) {
//             Serial.print(dataToSend[i], HEX);
//             Serial.print(" ");
//         }
//         Serial.println();

//         byte *frameToSendPtr = create_serial_frame(dataToSend, dataLength);

//         byte checksum_value_debug = frameToSendPtr[dataLength + 1];
//         // Serial.print("Checksum Value (from frame): ");
//         // Serial.println(checksum_value_debug, HEX);


//         if (frameToSendPtr != NULL) {
//             // Serial.print("Sending Frame (Scalable Example Function): ");
//             for (size_t i = 0; i < dataLength + 3; i++) {
//                 Serial.print(frameToSendPtr[i], HEX);
//                 Serial.print(" ");
//             }
//             Serial.println();
//             Serial.write(frameToSendPtr, dataLength + 3);
//         } else {
//             Serial.println("Error: Frame creation failed (Scalable Example Function)!");
//         }
//     }
// }


// Define this globally or statically at the top of your sketch
#define send_max_data 21
byte lastSentData[send_max_data] = {0}; // To store the last sent data for comparison

void send_periodic_frame_example(bool send_flag) {
    static unsigned long lastSendTime = 0;
    unsigned long currentTime = millis();

    // Construct the current data to send
    byte dataToSend[send_max_data];
    dataToSend[0]  = collected_data.running_state;
    dataToSend[1]  = collected_data.wakeup_state;
    dataToSend[2]  = collected_data.communication_mode;
    dataToSend[3]  = collected_data.running_mode;
    dataToSend[4]  = collected_data.vlc_protocol_mode;
    dataToSend[5]  = collected_data.vlc_interval_seconds;
    dataToSend[6]  = collected_data.vlc_communication_volume;
    dataToSend[7]  = collected_data.vlc_information_volume;
    dataToSend[8]  = collected_data.vlc_per_unit_time;
    dataToSend[9]  = collected_data.pwm_percentage;
    dataToSend[10] = collected_data.ble_protocol_mode;
    dataToSend[11] = collected_data.ble_interval_seconds;
    dataToSend[12] = collected_data.ble_communication_volume;
    dataToSend[13] = collected_data.ble_information_volume;
    dataToSend[14] = collected_data.ble_per_unit_time;
    dataToSend[15] = collected_data.phy_rate_percentage;
    dataToSend[16] = collected_data.mtu_value;
    dataToSend[17] = collected_data.fast_time;
    dataToSend[18] = collected_data.spare_2;
    dataToSend[19] = collected_data.spare_3;
    dataToSend[20] = collected_data.spare_4;
    dataToSend[21] = 0;

    // Check if data has changed
    bool changed = false;
    for (int i = 0; i < send_max_data; i++) {
        if (dataToSend[i] != lastSentData[i]) {
            changed = true;
            break;
        }
    }

    if (send_flag || changed || (currentTime - lastSendTime >= 1000)) { // Only send if data changed or 1s passed
        lastSendTime = currentTime;

        // Save new data as last sent
        memcpy(lastSentData, dataToSend, sizeof(dataToSend));

        // Optional: Include timestamp as extra byte (if needed outside the 20-byte spec)
        // Otherwise, you can log/print it without sending

        // DEBUG print raw data
        // Serial.print("Sending Changed Data: ");
        // for (int i = 0; i < send_max_data; i++) {
        //     Serial.print(dataToSend[i], HEX);
        //     Serial.print(" ");
        // }
        // Serial.println();

        // Create frame and send
        byte* frameToSendPtr = create_serial_frame(dataToSend, send_max_data);
        if (frameToSendPtr != NULL) {
            Serial.write(frameToSendPtr, send_max_data+3); // 20 data + header + checksum + end
        } else {
            Serial.println("Frame creation failed.");
        }
    }
}


byte* create_serial_frame(byte* data_bytes, size_t data_length) {
  static byte frame[256]; // Static array to hold the frame
  if (data_length > 256 - 3) {
    Serial.println("Error: Data length exceeds maximum frame size.");
    return NULL;
  }

  frame[0] = HEADER_BYTE; // Header byte

  for (size_t i = 0; i < data_length; i++) {
    frame[i + 1] = data_bytes[i]; // Copy data bytes from input array
  }

  // Calculate checksum (sum of data bytes modulo 256)
  byte checksum_value = 0;
  for (size_t i = 0; i < data_length; i++) {
    checksum_value += data_bytes[i];
  }
  frame[data_length + 1] = checksum_value % 256; // Checksum byte
  frame[data_length + 2] = END_BYTE;             // End byte

  // --- DEBUG PRINT inside create_serial_frame ---
  // Serial.print("Calculated Checksum (inside create_serial_frame): "); // Print checksum calculation
  // Serial.println(checksum_value % 256, HEX);

  return frame; // Return pointer to the frame array
}