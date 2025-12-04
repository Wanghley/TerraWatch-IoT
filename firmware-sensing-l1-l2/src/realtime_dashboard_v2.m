% -------------------------------
% TerraWatch Real-Time Sensor Dashboard v2
% Optimized for RAW DATA COLLECTION MODE
%
% REQUIRES:
% 1. Octave 'instrument-control' package: pkg install -forge instrument-control
% 2. 'jsonlab' library: Download from FileExchange, place this script in its folder.
% 3. ESP32 firmware running in MODE_RAW_DATA_COLLECTION
% -------------------------------

clear all;
clc;
close all;

% --- Add jsonlab to path ---
addpath(pwd);
addpath(fullfile(pwd, 'jsonlab'));

% --- Load required packages ---
pkg load instrument-control

% --- Serial Port Configuration ---
port = "/dev/cu.usbmodem5A450483901";  % <-- CHANGE THIS to your ESP32's serial port
baud = 115200;  % MUST match firmware RAW DATA COLLECTION MODE (921600)

fprintf("\n╔═══════════════════════════════════════════════════════╗\n");
fprintf("║   TerraWatch Real-Time Dashboard (RAW DATA MODE)     ║\n");
fprintf("║                                                       ║\n");
fprintf("║   Serial Port: %s @ %d baud                   ║\n", port, baud);
fprintf("║                                                       ║\n");
fprintf("║   Commands to send via Serial:                       ║\n");
fprintf("║   - 'STOP'        : Stop data collection             ║\n");
fprintf("║   - 'RESET'       : Reset timing                     ║\n");
fprintf("║   - 'RATE <ms>'   : Change sample rate (e.g. 20)    ║\n");
fprintf("╚═══════════════════════════════════════════════════════╝\n\n");

% --- Plotting Buffers ---
HISTORY_LENGTH = 100;
radar_range_history_L = NaN(1, HISTORY_LENGTH);
radar_range_history_R = NaN(1, HISTORY_LENGTH);
radar_speed_history_L = NaN(1, HISTORY_LENGTH);
radar_speed_history_R = NaN(1, HISTORY_LENGTH);
radar_energy_history_L = NaN(1, HISTORY_LENGTH);
radar_energy_history_R = NaN(1, HISTORY_LENGTH);
mic_history_L = NaN(1, HISTORY_LENGTH);
mic_history_R = NaN(1, HISTORY_LENGTH);

% --- Prepare Figure ---
fig = figure('Name', 'TerraWatch Real-Time Dashboard', 'NumberTitle', 'off', 'WindowState', 'maximized');
set(fig, 'DefaultTextColor', 'black');

% --- Plot 1: Thermal Heatmap (8x24) ---
subplot(2, 3, [1 4]);  % Left side, both rows
thermal_matrix = zeros(8, 24);
h_thermal = imagesc(thermal_matrix, [20 35]);
colorbar('peer', gca);
axis equal tight;
title('Thermal Heatmap (8×24) — Left | Center | Right', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Pixel Column');
ylabel('Pixel Row');

% --- Plot 2: Radar Range ---
subplot(2, 3, 2);  % Top-middle
h_radar_range = plot(1:HISTORY_LENGTH, radar_range_history_L, 'b-', 'LineWidth', 1.5, ...
                     1:HISTORY_LENGTH, radar_range_history_R, 'r-', 'LineWidth', 1.5);
title('Radar Range (Last 100 cycles)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Cycle');
ylabel('Range (cm)');
ylim([0 2500]);
legend('R1 (Left)', 'R2 (Right)', 'Location', 'northeastoutside');
grid on;

% --- Plot 3: Radar Speed ---
subplot(2, 3, 5);  % Bottom-middle
h_radar_speed = plot(1:HISTORY_LENGTH, radar_speed_history_L, 'b--', 'LineWidth', 1.5, ...
                     1:HISTORY_LENGTH, radar_speed_history_R, 'r--', 'LineWidth', 1.5);
title('Radar Speed (Last 100 cycles)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Cycle');
ylabel('Speed (m/s)');
ylim auto;
legend('R1 (Left)', 'R2 (Right)', 'Location', 'northeastoutside');
grid on;

% --- Plot 4: Microphone ---
subplot(2, 3, [3 6]);  % Right side, both rows
h_mic = plot(1:HISTORY_LENGTH, mic_history_L, 'g-', 'LineWidth', 1.5, ...
             1:HISTORY_LENGTH, mic_history_R, 'm-', 'LineWidth', 1.5);
title('Microphone Levels (Last 100 cycles)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Cycle');
ylabel('Amplitude');
ylim auto;
legend('Left Mic', 'Right Mic', 'Location', 'northeastoutside');
grid on;

% --- Status Text (on last subplot) ---
h_status = text(0.5, 0.95, 'Status: Connecting...', 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'center', ...
    'BackgroundColor', 'yellow', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Interpreter', 'none');

drawnow;

% --- Connect to Serial ---
try
    s = serialport(port, baud);
    configureTerminator(s, "LF");  % Line terminator
    s.Timeout = 2;  % 2 second timeout

    fprintf("✓ Connected at %d baud\n", baud);
    fprintf("✓ Waiting for ESP32 data stream...\n\n");

    set(h_status, 'String', sprintf('Connected @ %d baud | Waiting for data...', baud), ...
        'BackgroundColor', [0.5 1 0.5]);
    pause(0.5);

catch err
    fprintf("✗ Failed to open serial port %s\n", port);
    fprintf("  Error: %s\n\n", err.message);
    error(sprintf("Cannot open %s. Check port name and device connection.", port));
end

% --- Real-Time Data Stream Loop ---
fprintf("Starting real-time data streaming...\n");
fprintf("═══════════════════════════════════════════════════════\n\n");

packet_count = 0;
error_count = 0;
last_status_update = tic;
start_time = tic;
first_good_packet = false;  % Track first successful parse

while ishandle(fig)
    try
        % Non-blocking check for available data
        if s.NumBytesAvailable > 0
            line = readline(s);

            % Only process JSON lines (start with '{')
            if ~isempty(line) && line(1) == '{'
                try
                    % Parse JSON
                    data = loadjson(line);
                    packet_count = packet_count + 1;

                    % DEBUG: Show first successful packet
                    if ~first_good_packet
                        fprintf("\n✓ First valid packet parsed successfully!\n");
                        fprintf("  Thermal: %d, %d, %d elements\n", ...
                            numel(data.thermal.left), numel(data.thermal.center), numel(data.thermal.right));
                        first_good_packet = true;
                    end

                    % ===== UPDATE THERMAL HEATMAP =====
                    if isfield(data, 'thermal')
                        l = data.thermal.left;
                        c = data.thermal.center;
                        r = data.thermal.right;

                        if numel(l) == 64 && numel(c) == 64 && numel(r) == 64
                            % Data is already numeric, just convert to double and reshape
                            l_data = double(l(:));
                            c_data = double(c(:));
                            r_data = double(r(:));

                            % Reshape to 8x8 grids
                            l_mat = reshape(l_data, [8, 8])';
                            c_mat = reshape(c_data, [8, 8])';
                            r_mat = reshape(r_data, [8, 8])';

                            % Combine into 8x24 matrix
                            thermal_matrix = [l_mat, c_mat, r_mat];

                            % Update heatmap
                            set(h_thermal, 'CData', thermal_matrix);
                        end
                    end

                    % ===== UPDATE RADAR PLOTS =====
                    if isfield(data, 'mmWave')
                        if isfield(data.mmWave, 'R1') && isfield(data.mmWave, 'R2')
                            % Extract range, speed, energy
                            r1_range = double(data.mmWave.R1.range_cm);
                            r2_range = double(data.mmWave.R2.range_cm);
                            r1_speed = double(data.mmWave.R1.speed_ms);
                            r2_speed = double(data.mmWave.R2.speed_ms);
                            r1_energy = double(data.mmWave.R1.energy);
                            r2_energy = double(data.mmWave.R2.energy);

                            % Shift and update range history
                            radar_range_history_L = [radar_range_history_L(2:end), r1_range];
                            radar_range_history_R = [radar_range_history_R(2:end), r2_range];

                            % Shift and update speed history
                            radar_speed_history_L = [radar_speed_history_L(2:end), r1_speed];
                            radar_speed_history_R = [radar_speed_history_R(2:end), r2_speed];

                            % Shift and update energy history (stored for potential future use)
                            radar_energy_history_L = [radar_energy_history_L(2:end), r1_energy];
                            radar_energy_history_R = [radar_energy_history_R(2:end), r2_energy];

                            % Update plots
                            set(h_radar_range(1), 'YData', radar_range_history_L);
                            set(h_radar_range(2), 'YData', radar_range_history_R);
                            set(h_radar_speed(1), 'YData', radar_speed_history_L);
                            set(h_radar_speed(2), 'YData', radar_speed_history_R);
                        end
                    end

                    % ===== UPDATE MICROPHONE PLOT =====
                    if isfield(data, 'mic')
                        mic_l = double(data.mic.left);
                        mic_r = double(data.mic.right);

                        % Shift and update history
                        mic_history_L = [mic_history_L(2:end), mic_l];
                        mic_history_R = [mic_history_R(2:end), mic_r];

                        % Update plot
                        set(h_mic(1), 'YData', mic_history_L);
                        set(h_mic(2), 'YData', mic_history_R);
                    end

                    % ===== UPDATE STATUS BAR =====
                    elapsed_status = toc(last_status_update);
                    if elapsed_status > 0.2  % Update every 200ms
                        total_elapsed = toc(start_time);
                        avg_freq = packet_count / total_elapsed;
                        status_str = sprintf('✓ %d packets | %d errors | %.1f Hz | %.1fs', ...
                            packet_count, error_count, avg_freq, total_elapsed);
                        set(h_status, 'String', status_str, 'BackgroundColor', [0.2 1 0.2]);
                        drawnow();
                        last_status_update = tic;
                    end

                catch parse_err
                    % JSON parsing failed
                    error_count = error_count + 1;
                    if error_count <= 3  % Print first 3 errors only
                        fprintf("\n✗ JSON Parse Error #%d:\n", error_count);
                        fprintf("  Error: %s\n", parse_err.message);
                        fprintf("  Line (first 80 chars): %s\n", line(1:min(80, length(line))));
                    end
                end
            end
        else
            % No data available, small pause to prevent busy-waiting
            pause(0.001);
        end

    catch loop_err
        % Serial error
        error_count = error_count + 1;
        set(h_status, 'String', '✗ Serial Error - Attempting to reconnect...', ...
            'BackgroundColor', [1 0.5 0.5]);

        fprintf("Serial error: %s\n", loop_err.message);

        % Check if port is still valid
        if isempty(s) || ~isvalid(s)
            fprintf("✗ Serial port disconnected.\n");
            break;
        end

        pause(0.5);
    end
end

% --- Cleanup ---
fprintf("\n═══════════════════════════════════════════════════════\n");
fprintf("✓ Dashboard closed\n");
fprintf("  Total packets: %d\n", packet_count);
fprintf("  Total errors: %d\n", error_count);
fprintf("  Runtime: %.1f seconds\n", toc(start_time));
fprintf("═══════════════════════════════════════════════════════\n\n");

try
    clear s;
catch
    % Serial already closed
end

disp("Dashboard stopped.");
