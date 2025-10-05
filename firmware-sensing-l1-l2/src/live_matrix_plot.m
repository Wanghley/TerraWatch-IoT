% Load the serial package
pkg load instrument-control

% --- Configuration ---
% Replace with your Arduinos serial port and baud rate
SERIAL_PORT = "/dev/ttyACM0"; % Linux/macOS example; use "COM3" on Windows
BAUD_RATE = 9600;

% Dimensions of your matrix
ROWS = 8;
COLS = 8;

% --- Setup ---
% Open the serial connection
s = serial(SERIAL_PORT, BAUD_RATE);
srl_flush(s); % Clear any old data in the buffer

printf("Connected to ESP-32. Starting plot. Press Ctrl+C to stop.\n");

% --- Main loop ---
while (true)
  try
    % Read multiple lines from the serial port
    serial_data = srl_read(s, ROWS * (COLS * 2 + 1), "char"); % Read bytes for a 3x3 matrix plus commas and newlines
    
    % Convert the string to a matrix using a string stream
    data_stream = strstream(serial_data);
    my_matrix = csvread(data_stream, 0, 0, [0, 0, ROWS-1, COLS-1]);

    % Plot the matrix as a surface
    surf(my_matrix);
    title("Live Matrix Data from ESP-32");
    xlabel("Column Index");
    ylabel("Row Index");
    zlabel("Value");
    pause(0.1); % Pause briefly to allow plot to update
    
  catch
    printf("An error occurred. Retrying...\n");
    srl_flush(s);
  end
end

% --- Cleanup ---
srl_close(s); % Close the serial port when done
printf("Serial connection closed.\n");
