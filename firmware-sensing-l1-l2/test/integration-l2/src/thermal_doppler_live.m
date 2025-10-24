pkg load instrument-control

% -------------------------------
% Serial port settings
% -------------------------------
port = "/dev/cu.usbmodem5A450483901";  % update if needed
baud = 115200;

s = serialport(port, baud);
configureTerminator(s, "LF");  % '\n'

disp("Starting real-time thermal + Doppler viewer...");
pause(2);

% -------------------------------
% Figure setup
% -------------------------------
figure('Name','Thermal + Doppler','NumberTitle','off');

% Thermal subplot (top, 80%)
ax1 = subplot(10,1,1:8);  % 8/10 of figure
thermal_matrix = zeros(8,8);
h_thermal = imagesc(ax1, thermal_matrix, [20 35]);
colorbar(ax1);
axis(ax1,'equal','tight');
xlabel(ax1,'X'); ylabel(ax1,'Y');
title(ax1,'Thermal Heatmap (°C)');

% -------------------------------
% Doppler subplots (bottom, 20% height, three columns)
ax_speed  = subplot('Position',[0.05,0.05,0.27,0.18]); hold(ax_speed,'on'); grid(ax_speed,'on');
ax_range  = subplot('Position',[0.37,0.05,0.27,0.18]); hold(ax_range,'on'); grid(ax_range,'on');
ax_energy = subplot('Position',[0.69,0.05,0.27,0.18]); hold(ax_energy,'on'); grid(ax_energy,'on');

% Set titles & labels once
title(ax_speed,'Doppler Speed'); xlabel(ax_speed,'Sample'); ylabel(ax_speed,'Speed (m/s)');
title(ax_range,'Doppler Range'); xlabel(ax_range,'Sample'); ylabel(ax_range,'Range (m)');
title(ax_energy,'Doppler Energy'); xlabel(ax_energy,'Sample'); ylabel(ax_energy,'Energy');

% Initialize line plots
h_speed  = plot(ax_speed, nan, nan, 'r');
h_range  = plot(ax_range, nan, nan, 'g');
h_energy = plot(ax_energy, nan, nan, 'b');

% Initialize data arrays
speed_data  = [];
range_data  = [];
energy_data = [];
time_data   = [];

% -------------------------------
% Moving average function
% -------------------------------
window_size = 3;
moving_avg = @(data) mean(data(max(1,end-window_size+1):end));

% -------------------------------
% Real-time loop
% -------------------------------
sample_count = 0;

while ishandle(h_thermal)
    while s.NumBytesAvailable > 0
        line = readline(s);

        try
            % --- Parse thermal ---
            line_clean = regexprep(line, '.*"thermal":\[(.*)\].*', '$1');
            nums = str2num(line_clean); %#ok<ST2NM>
            if numel(nums) == 64
                thermal_matrix = reshape(nums,[8,8]);
                set(h_thermal,'CData',thermal_matrix);
                title(ax1,sprintf('Thermal Heatmap (mean %.2f °C)', mean(thermal_matrix(:))));
                drawnow;
            end

            % --- Parse Doppler ---
            doppler_match = regexp(line, '"doppler":\{(.*?)\}', 'tokens');
            if ~isempty(doppler_match)
                doppler_str = doppler_match{1}{1};
                speed = str2double(regexp(doppler_str,'"speed":([0-9.-]+)','tokens','once'));
                range = str2double(regexp(doppler_str,'"range":([0-9.-]+)','tokens','once'));
                energy = str2double(regexp(doppler_str,'"energy":([0-9.-]+)','tokens','once'));

                % Clamp Range/Energy >=0
                range = max(0, range);
                energy = max(0, energy);

                % Update arrays
                sample_count += 1;
                speed_data(sample_count)  = speed;
                range_data(sample_count)  = range;
                energy_data(sample_count) = energy;
                time_data(sample_count)   = sample_count;

                % Apply simple moving average
                if sample_count >= window_size
                    speed_data(sample_count) = mean(speed_data(sample_count-window_size+1:sample_count));
                    range_data(sample_count) = mean(range_data(sample_count-window_size+1:sample_count));
                    energy_data(sample_count)= mean(energy_data(sample_count-window_size+1:sample_count));
                end

                % Update Doppler plots
                set(h_speed,'XData',time_data,'YData',speed_data);
                set(h_range,'XData',time_data,'YData',range_data);
                set(h_energy,'XData',time_data,'YData',energy_data);

                drawnow;
            end

        catch
            fprintf('⚠ Failed to parse line: %s\n',line);
        end
    end
    pause(0.01);
end

% Cleanup
clear s;
disp("Serial connection closed.");

