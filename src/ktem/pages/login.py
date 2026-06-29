import hashlib
import secrets
from pathlib import Path

import gradio as gr
from ktem.app import BasePage
from ktem.db.models import User, engine
from ktem.pages.resources.user import create_user
from sqlmodel import Session, select

GUEST_USERNAME = "guest"

fetch_creds = """
function() {
    const username = getStorage('username', '')
    const password = getStorage('password', '')
    return [username, password, null];
}
"""

signin_js = """
function(usn, pwd) {
    setStorage('username', usn);
    if ((usn || '').trim().toLowerCase() === 'guest') {
        removeFromStorage('password');
    } else {
        setStorage('password', pwd);
    }
    return [usn, pwd];
}
"""

guest_signin_js = """
function() {
    setStorage('username', 'guest');
    removeFromStorage('password');
    return [];
}
"""


class LoginPage(BasePage):

    public_events = ["onSignIn"]

    def __init__(self, app):
        self._app = app
        self.on_building_ui()

    def on_building_ui(self):
        logo_path = (
            Path(__file__).resolve().parents[1] / "assets" / "img" / "logo.jpg"
        )
        gr.Image(
            value=str(logo_path),
            show_label=False,
            show_download_button=False,
            container=False,
            interactive=False,
            elem_id="login-logo",
        )
        gr.Markdown(f"# Welcome to {self._app.app_name}!")
        self.usn = gr.Textbox(label="Username", visible=False)
        self.pwd = gr.Textbox(label="Password", type="password", visible=False)
        self.btn_login = gr.Button("Login", visible=False, variant="primary")
        self.btn_guest = gr.Button(
            "Access as Guest", visible=False, variant="secondary"
        )

    def on_register_events(self):
        onSignIn = gr.on(
            triggers=[self.btn_login.click, self.pwd.submit],
            fn=self.login,
            inputs=[self.usn, self.pwd],
            outputs=[self._app.user_id, self.usn, self.pwd],
            show_progress="hidden",
            js=signin_js,
        ).then(
            self.toggle_login_visibility,
            inputs=[self._app.user_id],
            outputs=[self.usn, self.pwd, self.btn_login, self.btn_guest],
        )
        for event in self._app.get_event("onSignIn"):
            onSignIn = onSignIn.success(**event)

        onGuestSignIn = self.btn_guest.click(
            fn=self.guest_login,
            inputs=[],
            outputs=[self._app.user_id, self.usn, self.pwd],
            show_progress="hidden",
            js=guest_signin_js,
        ).then(
            self.toggle_login_visibility,
            inputs=[self._app.user_id],
            outputs=[self.usn, self.pwd, self.btn_login, self.btn_guest],
        )
        for event in self._app.get_event("onSignIn"):
            onGuestSignIn = onGuestSignIn.success(**event)

    def toggle_login_visibility(self, user_id):
        return (
            gr.update(visible=user_id is None),
            gr.update(visible=user_id is None),
            gr.update(visible=user_id is None),
            gr.update(visible=user_id is None),
        )

    def _on_app_created(self):
        onSignIn = self._app.app.load(
            self.login,
            inputs=[self.usn, self.pwd],
            outputs=[self._app.user_id, self.usn, self.pwd],
            show_progress="hidden",
            js=fetch_creds,
        ).then(
            self.toggle_login_visibility,
            inputs=[self._app.user_id],
            outputs=[self.usn, self.pwd, self.btn_login, self.btn_guest],
        )
        for event in self._app.get_event("onSignIn"):
            onSignIn = onSignIn.success(**event)

    def on_subscribe_public_events(self):
        self._app.subscribe_event(
            name="onSignOut",
            definition={
                "fn": self.toggle_login_visibility,
                "inputs": [self._app.user_id],
                "outputs": [self.usn, self.pwd, self.btn_login, self.btn_guest],
                "show_progress": "hidden",
            },
        )

    def guest_login(self):
        """Return the reserved, non-admin guest account, creating it if needed."""
        with Session(engine) as session:
            guest = session.exec(
                select(User).where(User.username_lower == GUEST_USERNAME)
            ).first()

        if guest is None:
            # The password is intentionally not a public demo credential. Guest access
            # is granted only through the restricted guest flow and can be revoked by
            # removing the account or disabling user management.
            create_user(
                GUEST_USERNAME,
                secrets.token_urlsafe(32),
                is_admin=False,
            )
            with Session(engine) as session:
                guest = session.exec(
                    select(User).where(User.username_lower == GUEST_USERNAME)
                ).first()

        if guest is None:
            raise gr.Error("Guest access could not be initialized")
        return guest.id, "", ""

    def login(self, usn, pwd, request: gr.Request):
        try:
            import gradiologin as grlogin

            user = grlogin.get_user(request)
        except (ImportError, AssertionError):
            user = None

        if user:
            user_id = user["sub"]
            with Session(engine) as session:
                stmt = select(User).where(
                    User.id == user_id,
                )
                result = session.exec(stmt).all()

            if result:
                print("Existing user:", user)
                return user_id, "", ""
            else:
                print("Creating new user:", user)
                create_user(
                    usn=user["email"],
                    pwd="",
                    user_id=user_id,
                    is_admin=False,
                )
                return user_id, "", ""
        else:
            if (usn or "").lower().strip() == GUEST_USERNAME:
                return self.guest_login()

            if not usn or not pwd:
                return None, usn, pwd

            hashed_password = hashlib.sha256(pwd.encode()).hexdigest()
            with Session(engine) as session:
                stmt = select(User).where(
                    User.username_lower == usn.lower().strip(),
                    User.password == hashed_password,
                )
                result = session.exec(stmt).all()
                if result:
                    return result[0].id, "", ""

                gr.Warning("Invalid username or password")
                return None, usn, pwd
