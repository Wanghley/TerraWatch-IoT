% --- Configuration ---
port = "/dev/cu.usbmodem5A450483901";  % Change to your ESP32 port (e.g. "COM3" on Windows)
baud = 115200;
npoints = 200;  % number of points visible in the plot window

% --- Open serial connection ---
sp = serialport(port, baud);
configureTerminator(sp, "LF");

fprintf("Listening to %s at %d baud...\n", port, baud);

data = zeros(1, npoints);
t = 1:npoints;

figure;
h = plot(t, data, 'b-', 'LineWidth', 1.5);
ylim([0 4]);
xlabel('Samples');
ylabel('RMS * 100');
title('Real-Time RMS Signal from ESP32 I2S');
grid on;

% --- Live read loop ---
while true
  try
    line = readline(sp);              % read one line
    vals = str2num(line);             %#ok<ST2NM>
    if numel(vals) == 3
      val = vals(3);                  % take last value
      data = [data(2:end), val];      % append new, drop oldest
      set(h, 'YData', data);
      drawnow limitrate;
    endif
  catch err
    warning("Error: %s", err.message);
    pause(0.1);
  end_try_catch
endwhile
