"""Send an email when a study finishes.

A full sweep is six architectures x (15 search trials + 3 configs x 5 runs) and takes
hours, so the machine is not worth sitting in front of. This mails the final report -
or the traceback, if the sweep died - to whoever is configured below.

Credentials are never stored in the repository. They are read from the environment:

    NOTIFY_SMTP_USER        the account the mail is sent *from*
    NOTIFY_SMTP_PASSWORD    that account's password
    NOTIFY_EMAIL_TO         recipient (default: DEFAULT_RECIPIENT below)
    NOTIFY_SMTP_HOST        default smtp.gmail.com
    NOTIFY_SMTP_PORT        default 587 (STARTTLS; 465 is used as implicit SSL)

or, so they do not have to be exported in every new shell, from a file of the same
KEY=VALUE lines at studies/notify_credentials.env (gitignored - do not commit it):

    NOTIFY_SMTP_USER=you@gmail.com
    NOTIFY_SMTP_PASSWORD=abcdefghijklmnop

Gmail specifically will not accept an account password over SMTP. Turn on 2-Step
Verification and generate a 16 character App Password
(https://myaccount.google.com/apppasswords), and use that as NOTIFY_SMTP_PASSWORD.

Real environment variables win over the file, so a one-off run can override it.
"""
import html
import os
import smtplib
import socket
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Optional, Tuple

STUDIES_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = STUDIES_DIR / "notify_credentials.env"

DEFAULT_RECIPIENT = "tekutisharijs@gmail.com"
DEFAULT_HOST = "smtp.gmail.com"
DEFAULT_PORT = 587
# SMTP can block for a long time on a bad host. A finished sweep must not be held hostage
# by the notification about it.
TIMEOUT_SECONDS = 30


def _read_credentials_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse KEY=VALUE lines. Missing file is not an error - the env may carry them."""
    # Resolved per call rather than as a default argument, which would freeze the module
    # constant at import time and make the location impossible to point elsewhere.
    path = path or CREDENTIALS_FILE
    if not path.exists():
        return {}

    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


def get_config() -> Dict[str, str]:
    """Resolve the mail settings, environment first, then the credentials file."""
    from_file = _read_credentials_file()

    def setting(name: str, default: str = "") -> str:
        return os.environ.get(name) or from_file.get(name, default)

    # Google displays App Passwords as "abcd efgh ijkl mnop" and the spaces get pasted in
    # with them, but SMTP AUTH wants the bare 16 characters and rejects the spaced form.
    password = "".join(setting("NOTIFY_SMTP_PASSWORD").split())

    return {
        "user": setting("NOTIFY_SMTP_USER"),
        "password": password,
        "to": setting("NOTIFY_EMAIL_TO", DEFAULT_RECIPIENT),
        "host": setting("NOTIFY_SMTP_HOST", DEFAULT_HOST),
        "port": setting("NOTIFY_SMTP_PORT", str(DEFAULT_PORT)),
    }


def check_config(config: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """Is sending possible? Called *before* the sweep starts, not after.

    Discovering that the password was never set is useless six hours later, when the
    mail is the only thing that was going to tell you the run had ended.
    """
    config = config or get_config()

    missing = [
        name for name, key in (("NOTIFY_SMTP_USER", "user"), ("NOTIFY_SMTP_PASSWORD", "password"))
        if not config[key]
    ]
    if missing:
        return False, (
            f"email notification is off: {' and '.join(missing)} not set.\n"
            f"  Set them in the environment or in {CREDENTIALS_FILE}\n"
            f"  (for Gmail the password must be an App Password, not the account password:\n"
            f"   https://myaccount.google.com/apppasswords)"
        )

    if not config["to"]:
        return False, "email notification is off: NOTIFY_EMAIL_TO is empty."

    try:
        int(config["port"])
    except ValueError:
        return False, f"email notification is off: NOTIFY_SMTP_PORT={config['port']!r} is not a number."

    return True, f"email notification on: {config['to']} via {config['host']}:{config['port']}"


def send_email(subject: str, body: str, config: Optional[Dict[str, str]] = None) -> bool:
    """Send one plain text mail. Returns whether it went out.

    Never raises: the sweep's results are already on disk by the time this is called, so
    a mail server problem is worth a warning and nothing more.
    """
    config = config or get_config()

    usable, reason = check_config(config)
    if not usable:
        print(f"\n[notify] not sent - {reason}")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["user"]
    message["To"] = config["to"]
    message.set_content(body)
    # The body is a column-aligned table. Gmail renders text/plain in a proportional font,
    # which turns it into a mess on a phone, so the same text goes out as a monospaced
    # <pre> alternative; clients that prefer plain text still get the version above.
    message.add_alternative(
        f"<pre style=\"font-family: ui-monospace, Menlo, Consolas, monospace; "
        f"font-size: 12px; white-space: pre;\">{html.escape(body)}</pre>",
        subtype="html",
    )

    port = int(config["port"])
    try:
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(config["host"], port, timeout=TIMEOUT_SECONDS, context=context) as server:
                server.login(config["user"], config["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(config["host"], port, timeout=TIMEOUT_SECONDS) as server:
                server.starttls(context=context)
                server.login(config["user"], config["password"])
                server.send_message(message)
    except smtplib.SMTPAuthenticationError:
        print(
            f"\n[notify] not sent - {config['host']} rejected the login for {config['user']}.\n"
            f"  For Gmail, NOTIFY_SMTP_PASSWORD has to be a 16 character App Password\n"
            f"  (https://myaccount.google.com/apppasswords), not the account password."
        )
        return False
    except (smtplib.SMTPException, OSError) as error:
        # OSError covers DNS failure, refused connection and the socket timeout.
        print(f"\n[notify] not sent - {type(error).__name__}: {error}")
        return False

    print(f"\n[notify] emailed {config['to']}: {subject}")
    return True


def machine_line() -> str:
    """Where the run happened - useful when the mail arrives from a lab machine."""
    return f"host {socket.gethostname()}, pid {os.getpid()}"


if __name__ == "__main__":
    # `python studies/notify.py` sends one test mail, so the credentials can be proven
    # in a few seconds rather than by starting a sweep and finding out hours later.
    config = get_config()
    usable, reason = check_config(config)
    print(reason)
    if usable:
        sent = send_email(
            "[Exercise recognition] test message",
            "If this arrived, the study scripts can email you when a run finishes.\n\n"
            f"Sent from {machine_line()}.",
            config,
        )
        raise SystemExit(0 if sent else 1)
    raise SystemExit(1)
