# FaceLog

## Project Description

FaceLog is a desktop face recognition and registration application built with Python, PyQt5, Qt Designer, OpenCV, face_recognition, and PostgreSQL. The application allows users to register personal information, capture a face image through the computer camera, store face data in a PostgreSQL database, and later identify registered users through facial recognition.

This project is intended to run as a Windows desktop application. The user interface is designed using Qt Designer `.ui` files and loaded by the Python application at runtime.

## Key Features

- Desktop Qt Interface: Uses PyQt5 and Qt Designer UI files for the login, registration, and user screens.
- Face Registration: Captures a user's face through the default camera and stores the face image and encoding.
- Face Recognition Login: Scans faces in real time and compares them with registered face data from the database.
- PostgreSQL Storage: Saves user profile details, face encodings, and face images in a local PostgreSQL database.
- User Validation: Checks required registration fields and validates details such as phone number and email.
- Review Screen: Allows users to review registration details before saving them.

## Technology Stack

- Python
- PyQt5
- Qt Designer
- OpenCV
- face_recognition
- NumPy
- PostgreSQL
- psycopg2

## Project Files

- `main.py` - Main application file.
- `login.ui` - Qt Designer UI file for the login screen.
- `user.ui` - Qt Designer UI file for the user, registration, and review screens.
- `resources.qrc` - Qt resource collection file.
- `resources_rc.py` - Compiled Qt resource file used by the application.
- `images/` - Image and icon assets used by the UI.

## Requirements

Before running the application, make sure the following are installed:

- Python 3.10 or newer
- PostgreSQL
- A working webcam
- Visual C++ Build Tools may be required for `face_recognition` and `dlib`

Install the Python dependencies:

```powershell
pip install PyQt5 opencv-python psycopg2-binary numpy face-recognition
```

## Database Setup

The application connects to PostgreSQL using these settings:

```text
host: localhost
port: 5432
database: Facelog
user: postgres
password: password
```

Make sure PostgreSQL is running before opening the application.

You can test the database connection with:

```powershell
psql -U postgres -d Facelog -p 5432
```

To view tables inside PostgreSQL:

```sql
\dt
```

To view the users table structure:

```sql
\d users
```

## Running the Application

From the project folder, run:

```powershell
python main.py
```

The application should open the login screen first.

## UI Notes

The UI was designed in Qt Designer with a main window size of approximately `1000 x 600` pixels.

Recommended display settings:

- Screen resolution: `1366 x 768` or higher
- Windows display scaling: `100%`

Because the current UI uses fixed widget positions, very small screens or high Windows scaling values may cause parts of the interface to appear clipped.

## Important Path Note

The application loads `.ui` files using `loadUi(...)`. If the application does not open correctly, check that `main.py` points to the correct locations of:

```text
login.ui
user.ui
```

For better portability, the UI files should be loaded from the same folder as `main.py`.

## Use Cases

- Registering users with face data for local identification.
- Desktop-based face recognition login or attendance workflows.
- Small-scale user record management with camera-based verification.

## Contributing

Contributions are welcome. If you make changes, keep the desktop PyQt5 workflow in mind and test the registration, camera, and database features before submitting updates.

## License

This project is licensed under the MIT License.
