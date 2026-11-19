from getpass import getpass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

import pulseflow.database.models  # noqa: F401
from pulseflow.core.database import get_engine
from pulseflow.core.security import hash_password
from pulseflow.database.models.hospital import Role, User, UserRole


def create_admin() -> None:
    full_name = input("Full name: ").strip()
    email = input("Email: ").strip().lower()

    if not full_name or len(full_name) > 150:
        raise ValueError("Full name must contain 1 to 150 characters.")

    if (
        len(email) > 255
        or email.count("@") != 1
        or any(character.isspace() for character in email)
        or not all(email.split("@"))
    ):
        raise ValueError("Enter a valid email address.")

    password = getpass("Password (at least 12 characters): ")
    confirmation = getpass("Confirm password: ")

    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")

    if password != confirmation:
        raise ValueError("Passwords do not match.")

    password_hash = hash_password(password)
    engine = get_engine()

    try:
        with Session(engine) as session:
            with session.begin():
                existing_user = session.scalar(
                    select(User).where(func.lower(User.email) == email)
                )

                if existing_user is not None:
                    raise ValueError(
                        "Email already exists. No account was changed."
                    )

                role = session.scalar(
                    select(Role).where(Role.name == "admin")
                )

                if role is None:
                    role = Role(name="admin")
                    session.add(role)
                    session.flush()

                user = User(
                    full_name=full_name,
                    email=email,
                    password_hash=password_hash,
                    is_active=True,
                )
                session.add(user)
                session.flush()

                session.add(UserRole(user_id=user.id, role_id=role.id))

        print("Admin account created successfully.")

    finally:
        engine.dispose()


if __name__ == "__main__":
    create_admin()