import getpass
from app import app, db, bcrypt, User

def create_admin_user():
    """
    Initializes the database and creates the first admin user.
    This script should be run once from the command line.
    """
    # app_context() is required to access the database outside of a web request
    with app.app_context():
        # Create all database tables based on the models defined in app.py
        # This will create the 'site.db' file and the 'user' table if they don't exist.
        print("Creating database tables...")
        db.create_all()
        print("Tables created successfully.")

        # Check if an admin already exists to prevent creating duplicates
        if User.query.filter_by(role='admin').first():
            print("An admin user already exists. Aborting.")
            return

        print("\n--- Create Your Admin Account ---")
        # Prompt for the admin's username
        username = input("Enter a username for the admin: ")
        
        # Prompt for the admin's password securely (it won't be visible as you type)
        password = getpass.getpass("Enter a password for the admin: ")
        
        # Hash the password for secure storage
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create the new user object with the 'admin' role
        admin_user = User(username=username, password_hash=hashed_password, role='admin')
        
        # Add the new admin user to the database session
        db.session.add(admin_user)
        # Commit the changes to permanently save the user to the database
        db.session.commit()
        
        print(f"\nAdmin user '{username}' was created successfully!")

if __name__ == '__main__':
    create_admin_user()