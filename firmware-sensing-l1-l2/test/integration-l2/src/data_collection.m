pkg load instrument-control  % for serial port communication

function data_collection(port, baud)
  % Default values
  if nargin < 1, port = "/dev/cu.usbmodem5A450483901"; endif
  if nargin < 2, baud = 115200; endif

  % --- Serial setup ---
  printf("Opening serial port %s at %d baud...\n", port, baud);
  s = serial(port, baud);
  s.timeout = 1;  % read timeout in seconds
  fopen(s);
  pause(5);  % give ESP32 time to reset/start

  % --- CSV file setup ---
  ts = strftime("%Y%m%d_%H%M%S", localtime(time()));
  filename = sprintf("dataset_%s.csv", ts);
  fid = fopen(filename, "w");

  % Write CSV header
  fprintf(fid, "sample,timestamp,label,doppler_speed,doppler_range,doppler_energy,mic_rms_mean,mic_peak_mean");
  for i = 1:64
    fprintf(fid, ",thermal_%d", i);
  endfor
  fprintf(fid, "\n");
  fflush(fid);

  % --- Wait for BEGIN DATA STREAM ---
  printf("Waiting for BEGIN DATA STREAM...\n");
  while true
    try
      line = strtrim(fgetl(s));  % read line with timeout
      if ~isempty(line)
        printf("%s\n", line);  % debug output
        if contains(line, "=== BEGIN DATA STREAM ===")
          printf("✅ Data stream started.\n");
          break;
        endif
      endif
    catch
      % ignore read timeout
    end_try_catch
  endwhile

  % --- Main collection loop ---
  sample_count = 0;
  printf("Collecting data...\n");

  while true
    try
      line = strtrim(fgetl(s));
      if isempty(line)
        continue;  % skip empty lines or timeouts
      endif

      % Stop on END marker
      if contains(line, "=== END DATA STREAM ===")
        printf("\n⏹ End of data stream detected.\n");
        break;
      endif

      % Parse JSON-like line
      if startsWith(line, "{")
        sample_count += 1;

        % Extract fields using regexp
        m = regexp(line, '"sample":(\d+)', 'tokens'); sample = str2double(m{1}{1});
        m = regexp(line, '"timestamp":(\d+)', 'tokens'); timestamp = str2double(m{1}{1});
        m = regexp(line, '"label":(\d+)', 'tokens'); label = str2double(m{1}{1});
        m = regexp(line, '"speed":([\d\.\-]+)', 'tokens'); doppler_speed = str2double(m{1}{1});
        m = regexp(line, '"range":([\d\.\-]+)', 'tokens'); doppler_range = str2double(m{1}{1});
        m = regexp(line, '"energy":([\d\.\-]+)', 'tokens'); doppler_energy = str2double(m{1}{1});
        m = regexp(line, '"rms_mean":([\d\.\-]+)', 'tokens'); mic_rms_mean = str2double(m{1}{1});
        m = regexp(line, '"peak_mean":(\d+)', 'tokens'); mic_peak_mean = str2double(m{1}{1});
        m = regexp(line, '"thermal":\[([\d\.,\s]+)\]', 'tokens');
        thermal_values = str2num(["[" m{1}{1} "]"]);

        % Write CSV line
        fprintf(fid, "%d,%d,%d,%.4f,%.3f,%.0f,%.5f,%d", ...
                sample, timestamp, label, doppler_speed, doppler_range, doppler_energy, mic_rms_mean, mic_peak_mean);
        for i = 1:length(thermal_values)
          fprintf(fid, ",%.2f", thermal_values(i));
        endfor
        fprintf(fid, "\n");
        fflush(fid);

        if mod(sample_count, 10) == 0
          printf(".");
        endif
      endif

    catch
      warning("Parse error or timeout on line: %s", line);
    end_try_catch
  endwhile

  % --- Cleanup ---
  fclose(fid);
  fclose(s);
  printf("\nSaved dataset to: %s\nTotal samples: %d\n", filename, sample_count);
endfunction

