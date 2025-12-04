#!/usr/bin/env octave
% TerraWatch Real-Time Dashboard v3 - SIMPLIFIED & WORKING
% Clean, reliable serial plotting for ESP32 sensor data

clear all; close all; clc;

% Add jsonlab to path
addpath(pwd);
addpath(fullfile(pwd, 'jsonlab'));
pkg load instrument-control;

% Configuration
port = "/dev/cu.usbmodem5A450483901";
baud = 115200;

fprintf("\n╔════════════════════════════════════════╗\n");
fprintf("║  TerraWatch Dashboard v3 - SIMPLIFIED ║\n");
fprintf("║  Port: %s @ %d baud        ║\n", port, baud);
fprintf("╚════════════════════════════════════════╝\n\n");

% Try to open serial port
try
    s = serialport(port, baud);
    configureTerminator(s, "LF");
    s.Timeout = 2;
    fprintf("✓ Connected to %s @ %d baud\n\n", port, baud);
catch err
    fprintf("✗ Failed to open port: %s\n", err.message);
    return;
end

% Create figure with simple layout
fig = figure('Name', 'TerraWatch Dashboard', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 900]);

% Plot 1: Thermal (left, wide)
ax1 = subplot(2, 2, 1);
im = imagesc(zeros(8, 24), [20 35]);
colorbar;
title('Thermal Heatmap (8×24)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Pixel'); ylabel('Row');

% Plot 2: Radar Range (top right)
ax2 = subplot(2, 2, 2);
hold on;
line_r1 = plot(1:100, NaN(1, 100), 'b-', 'LineWidth', 2, 'DisplayName', 'R1');
line_r2 = plot(1:100, NaN(1, 100), 'r-', 'LineWidth', 2, 'DisplayName', 'R2');
title('Radar Range (cm)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Sample'); ylabel('Range');
ylim([0 2500]);
legend('Location', 'northeast');
grid on;
hold off;

% Plot 3: Radar Speed + Microphone (bottom right)
ax3 = subplot(2, 2, 4);
hold on;
line_speed1 = plot(1:100, NaN(1, 100), 'b--', 'LineWidth', 2, 'DisplayName', 'Speed R1');
line_speed2 = plot(1:100, NaN(1, 100), 'r--', 'LineWidth', 2, 'DisplayName', 'Speed R2');
line_mic_l = plot(1:100, NaN(1, 100), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Mic L');
line_mic_r = plot(1:100, NaN(1, 100), 'm-', 'LineWidth', 1.5, 'DisplayName', 'Mic R');
title('Speed (m/s) & Mic Amplitude', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Sample'); ylabel('Value');
legend('Location', 'northeast');
grid on;
hold off;

% Title for right column (combined plot)
subplot(2, 2, 3);
axis off;

% Initialize buffers
thermal_buf = zeros(8, 24);
hist_len = 100;
range_hist = NaN(2, hist_len);
speed_hist = NaN(2, hist_len);
mic_hist = NaN(2, hist_len);

% Counters
packet_cnt = 0;
error_cnt = 0;
start_t = tic;
last_update = tic;

fprintf("Waiting for data...\n");

% Main loop
while ishandle(fig)
    try
        if s.NumBytesAvailable > 0
            line = readline(s);
            
            % Skip non-JSON lines
            if isempty(line) || line(1) ~= '{'
                continue;
            end
            
            try
                % Parse JSON
                data = loadjson(line);
                packet_cnt = packet_cnt + 1;
                
                % === THERMAL ===
                if isfield(data, 'thermal')
                    t = data.thermal;
                    if isfield(t, 'left') && isfield(t, 'center') && isfield(t, 'right')
                        l = double(t.left);
                        c = double(t.center);
                        r = double(t.right);
                        if numel(l)==64 && numel(c)==64 && numel(r)==64
                            lm = reshape(l, 8, 8)';
                            cm = reshape(c, 8, 8)';
                            rm = reshape(r, 8, 8)';
                            thermal_buf = [lm, cm, rm];
                            set(im, 'CData', thermal_buf);
                        end
                    end
                end
                
                % === RADAR ===
                if isfield(data, 'mmWave')
                    mm = data.mmWave;
                    if isfield(mm, 'R1') && isfield(mm, 'R2')
                        r1 = mm.R1; r2 = mm.R2;
                        
                        % Range
                        if isfield(r1, 'range_cm') && isfield(r2, 'range_cm')
                            range_hist = [range_hist(:, 2:end), [double(r1.range_cm); double(r2.range_cm)]];
                            set(line_r1, 'YData', range_hist(1, :));
                            set(line_r2, 'YData', range_hist(2, :));
                        end
                        
                        % Speed
                        if isfield(r1, 'speed_ms') && isfield(r2, 'speed_ms')
                            speed_hist = [speed_hist(:, 2:end), [double(r1.speed_ms); double(r2.speed_ms)]];
                            set(line_speed1, 'YData', speed_hist(1, :));
                            set(line_speed2, 'YData', speed_hist(2, :));
                        end
                    end
                end
                
                % === MICROPHONE ===
                if isfield(data, 'mic')
                    m = data.mic;
                    if isfield(m, 'left') && isfield(m, 'right')
                        mic_hist = [mic_hist(:, 2:end), [double(m.left); double(m.right)]];
                        set(line_mic_l, 'YData', mic_hist(1, :));
                        set(line_mic_r, 'YData', mic_hist(2, :));
                    end
                end
                
                % Update display every 2 seconds
                if toc(last_update) > 2.0
                    elapsed = toc(start_t);
                    freq = packet_cnt / elapsed;
                    fprintf('Packets: %d | Errors: %d | Freq: %.1f Hz | Time: %.1fs\n', ...
                        packet_cnt, error_cnt, freq, elapsed);
                    drawnow;
                    last_update = tic;
                end
                
            catch parse_err
                error_cnt = error_cnt + 1;
                if error_cnt <= 3
                    fprintf("✗ Parse error #%d: %s\n", error_cnt, parse_err.message);
                end
            end
        else
            pause(0.001);
        end
        
    catch loop_err
        fprintf("Loop error: %s\n", loop_err.message);
        if ~isvalid(s)
            fprintf("Port disconnected\n");
            break;
        end
    end
end

% Cleanup
try close(s); end
fprintf("\nDashboard closed. Total packets: %d, Errors: %d\n", packet_cnt, error_cnt);
