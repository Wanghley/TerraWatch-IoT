% -------------------------------
% Real-Time 8x24 Thermal Heatmap from Serial
% -------------------------------

pkg load instrument-control   % Required for serial support in Octave

% -------------------------------
% Serial Port Settings
% -------------------------------
port = "COM6";  % ← change to your serial device
baud = 115200;

% Open serial port
s = serialport(port, baud);
configureTerminator(s, "LF");   % '\n' line ending

disp("Starting real-time 8x24 heatmap viewer...");
pause(2);   % give port some time to stabilize

% -------------------------------
% Prepare figure
% -------------------------------
figure('Name','Thermal Camera 8x24 Heatmap','NumberTitle','off');
matrix = zeros(8,24);          % 8 rows x 24 columns (left-center-right)
h = imagesc(matrix, [20 35]);  % adjust [min max] to your temp range
colorbar;
axis equal tight;
xlabel('X'); ylabel('Y');
title('Live Thermal Camera Feed');
drawnow;

% -------------------------------
% Real-Time Loop
% -------------------------------
buffer = "";  % accumulated serial data

while ishandle(h)
    % Read any available serial data
    while s.NumBytesAvailable > 0
        line = readline(s);
        buffer = [buffer, line, "\n"];
    end

    % Find last closing bracket ']'
    pos = rindex(buffer, "]");
    if pos > 0
        frame = buffer(1:pos);
        buffer = buffer(pos+1:end);

        % Clean frame
        frame = strrep(frame, '[', '');
        frame = strrep(frame, ']', '');
        frame = strrep(frame, ',', ' ');
        frame = strtrim(frame);

        % Convert to numbers
        nums = str2num(frame); %#ok<ST2NM>

        if numel(nums) == 192   % 3 sensors × 64 pixels
            % Reshape row-wise into 8x24
            matrix = zeros(8,24);
            for row = 1:8
                leftRow   = nums((row-1)*8 + (1:8));
                centerRow = nums(64 + (row-1)*8 + (1:8));
                rightRow  = nums(128 + (row-1)*8 + (1:8));
                matrix(row,:) = [leftRow, centerRow, rightRow];
            end

            set(h, 'CData', matrix);
            title(sprintf('Live Thermal Camera Feed (mean %.2f °C)', mean(matrix(:))));
            drawnow;

        elseif numel(nums) > 0
            fprintf("⚠ Incomplete or bad frame (%d values)\n", numel(nums));
        end
    else
        pause(0.02);  % small wait when no full frame found
    end
end

% -------------------------------
% Cleanup
% -------------------------------
clear s;
disp("Serial connection closed.");
