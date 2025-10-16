% -------------------------------
% Real-Time 8x8 Thermal Heatmap from Serial (Stable Octave Version)
% -------------------------------

pkg load instrument-control   % Required for serial support in Octave

% -------------------------------
% Serial Port Settings
% -------------------------------
port = "/dev/cu.usbmodem5A450485831";  % ← change this to your serial device
baud = 115200;

% Open serial port (Octave ≥6)
s = serialport(port, baud);
configureTerminator(s, "LF");   % '\n' line ending

disp("Starting real-time 8x8 heatmap viewer...");
pause(2);   % give port some time to stabilize

% -------------------------------
% Prepare figure
% -------------------------------
figure('Name','Thermal Camera 8x8 Heatmap','NumberTitle','off');
matrix = zeros(8,8);
h = imagesc(matrix, [20 35]);   % adjust [min max] to your temp range
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
    pos = rindex(buffer, "]");   % works in Octave (no contains)
    if pos > 0
        % Extract everything up to that point as one full frame
        frame = buffer(1:pos);
        buffer = buffer(pos+1:end);

        % Clean frame
        frame = strrep(frame, '[', '');
        frame = strrep(frame, ']', '');
        frame = strrep(frame, ',', ' ');
        frame = strtrim(frame);

        % Convert to numbers
        nums = str2num(frame); %#ok<ST2NM>
        if numel(nums) == 64
            matrix = reshape(nums, [8,8]);
            set(h, 'CData', matrix);
            title(sprintf('Live Thermal Camera Feed (mean %.2f °C)', mean(matrix(:))));
            drawnow;   % simpler, Octave-compatible refresh
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

