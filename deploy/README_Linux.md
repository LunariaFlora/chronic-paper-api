# Running in the background on Linux

The service has been tested on a local Ubuntu / WSL2 machine with Python 3.12. The target server still needs the deploying party to install the dependencies and run one integration pass with the platform.

Foreground startup, inputs and outputs are described in the repository README. To keep the service running after the terminal closes, use the systemd templates in this directory:

- `chronic.service`: the service definition. The default program directory is `/opt/chronic` and the service runs as user and group `chronic`.
- `chronic.env.example`: copy it to `/etc/chronic.env` and fill in your own project token; the default data directory is `/var/lib/chronic`.
- The deploying party creates that account and data directory, makes the program readable and the data directory writable by the `chronic` user, and installs a `.venv` inside the program directory.
- Place the service definition at `/etc/systemd/system/chronic.service`, then run `sudo systemctl daemon-reload` and `sudo systemctl enable --now chronic`.
- Check the status with `systemctl status chronic` and the log with `journalctl -u chronic -n 50`.

Adjust the service definition when you use a different directory or account. For direct external access, set `CHRONIC_HOST=0.0.0.0` in the environment file and configure a token; when the service is reached through a reverse proxy on the same host, `127.0.0.1` can stay.

The shell and Python entry points were verified in this delivery. The systemd template and the Docker files are provided for the deploying party and were not installed on their server as part of this work.
