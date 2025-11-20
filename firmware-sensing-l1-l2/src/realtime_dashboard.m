% -------------------------------
% TerraWatch Real-Time Sensor Dashboard
%
% REQUIRES:
% 1. Octave 'instrument-control' package: pkg install -forge instrument-control
% 2. 'jsonlab' library: Download from FileExchange, place this script in its folder.
% -------------------------------

clear all;
clc;
close all;

% --- Add jsonlab to path ---
% Assumes this script is in the folder and jsonlab files are in the jsonlab subfolder
addpath(pwd);
addpath(fullfile(pwd, 'jsonlab'));

% --- Load required packages ---
pkg load instrument-control

% -------------------------------
% Serial Port Settings
% -------------------------------
port = "COM6";  % <-- CHANGE THIS to your ESP32's serial port
baud = 115200;

% -------------------------------
% Plotting Buffers
% -------------------------------
HISTORY_LENGTH = 100;
radar_range_history_L = NaN(1, HISTORY_LENGTH);
radar_range_history_R = NaN(1, HISTORY_LENGTH);
mic_history_L = NaN(1, HISTORY_LENGTH);
mic_history_R = NaN(1, HISTORY_LENGTH);

% -------------------------------
% Prepare Figure
% -------------------------------
fig = figure('Name','TerraWatch Sensor Dashboard','NumberTitle','off', 'WindowState', 'maximized');
set(fig, 'DefaultTextColor', 'black');

% --- Plot 1: Thermal Heatmap (8x24) ---
subplot(2, 2, [1 2]); % Top half of the figure
thermal_matrix = zeros(8, 24); % 8 rows x (8+8+8) cols
h_thermal = imagesc(thermal_matrix, [20 35]); % [20 35] = Temp range C
colorbar;
axis equal tight;
title('Thermal Heatmap (8x24)');
xlabel('Sensor Array (Left - Center - Right)');
ylabel('Pixel Row');

% --- Plot 2: Radar Time-Series ---
subplot(2, 2, 3); % Bottom-left
h_radar = plot(1:HISTORY_LENGTH, radar_range_history_L, 'b-', 1:HISTORY_LENGTH, radar_range_history_R, 'r-');
title('Radar Range (Last 100 cycles)');
xlabel('Time');
ylabel('Range (cm)');
ylim([0 10]); % 25m = 2500cm
legend('Left Radar', 'Right Radar');
grid on;

% --- Plot 3: Mic Time-Series ---
subplot(2, 2, 4); % Bottom-right
h_mic = plot(1:HISTORY_LENGTH, mic_history_L, 'g-', 1:HISTORY_LENGTH, mic_history_R, 'm-');
title('Microphone Levels (Last 100 cycles)');
xlabel('Time');
ylabel('Amplitude (RMS)');
ylim auto;
legend('Left Mic', 'Right Mic');
grid on;

drawnow;

% -------------------------------
% Connect to Serial
% -------------------------------
try
    s = serialport(port, baud);
    configureTerminator(s, "LF"); % '\n' line ending from Serial.println()
    disp("Successfully connected. Waiting for data...");
catch err
    disp(err.message);
    error("Failed to open serial port. Is it correct? Is the device plugged in?");
end

% -------------------------------
% Real-Time Loop
% -------------------------------
while ishandle(fig)
    try
        % Read one full line from serial (ends in \n)
        line = readline(s);

        % If line is not empty, try to parse it as JSON
        if ~isempty(line)
            % Use jsonlab's loadjson
            data = loadjson(line);

            % --- 1. Update Thermal Heatmap ---
            if isfield(data, 'thermal')
                % Get all 3 arrays
                l = data.thermal.left;
                c = data.thermal.center;
                r = data.thermal.right;

                % Check if data is valid
                if numel(l) == 64 && numel(c) == 64 && numel(r) == 64
                    % Reshape each 64-element array into an 8x8 matrix
                    % The ' transpose is needed because reshape is column-major
                    l_mat = reshape(l, [8, 8])';
                    c_mat = reshape(c, [8, 8])';
                    r_mat = reshape(r, [8, 8])';

                    % Combine into the final 8x24 matrix
                    thermal_matrix = [l_mat, c_mat, r_mat];

                    % Update the plot data
                    set(h_thermal, 'CData', thermal_matrix);
                end
            end

            % --- 2. Update Radar Plot ---
            if isfield(data, 'radar')
                % Get new values
                range_l = data.radar.left.range;
                range_r = data.radar.right.range;

                % Shift history buffers
                radar_range_history_L = [radar_range_history_L(2:end), range_l];
                radar_range_history_R = [radar_range_history_R(2:end), range_r];

                % Update plot data
                set(h_radar(1), 'YData', radar_range_history_L);
                set(h_radar(2), 'YData', radar_range_history_R);
            end

            % --- 3. Update Mic Plot ---
            if isfield(data, 'mic')
                % Get new values
                mic_l = data.mic.left;
                mic_r = data.mic.right;

                % Shift history buffers
                mic_history_L = [mic_history_L(2:end), mic_l];
                mic_history_R = [mic_history_R(2:end), mic_r];

                % Update plot data
                set(h_mic(1), 'YData', mic_history_L);
                set(h_mic(2), 'YData', mic_history_R);
            end

            % Redraw the figure
            drawnow;
        end

    catch err
        % Handle errors (e.g., incomplete JSON, serial disconnect)
        disp("Error during loop. Trying to continue...");
        disp(err.message);
        % Check if serial port is still open
        if isempty(s) || ~isvalid(s)
            disp("Serial port disconnected. Exiting.");
            break;
        end
        % Pause to avoid spamming errors
        pause(0.1);
    end
end

% -------------------------------
% Cleanup
% -------------------------------
clear s;
disp("Serial connection closed.");
