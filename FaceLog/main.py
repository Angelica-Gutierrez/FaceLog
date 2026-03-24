import sys
import cv2  # For camera access and face detection
import psycopg2  # For PostgreSQL database access
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import resources_rc  
from datetime import datetime 
from PyQt5.QtCore import QDate
import re
import face_recognition
import numpy as np
import io
from io import BytesIO

class LoginScreen(QMainWindow):
    def __init__(self):
        super(LoginScreen, self).__init__()
        loadUi(r"C:\Users\angelica\Downloads\FaceLog\login.ui", self)

        # Disable maximize button
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        # Connect buttons
        self.loginButton.clicked.connect(self.open_user_screen)
        self.registerButton.clicked.connect(self.open_registration)
        
    def open_user_screen(self):
        # Open the UserScreen UI and close the LoginScreen.
        self.user_screen = UserScreen()
        self.user_screen.show()
        self.close()

    def open_registration(self):
        # Switch to the registration screen directly from the LoginScreen.
        self.registration_screen = RegistrationScreen()
        self.registration_screen.show()
        self.close()    


class UserScreen(QMainWindow):
    def __init__(self):
        super(UserScreen, self).__init__()
        loadUi(r"C:\Users\angelica\Downloads\FaceLog\user.ui", self)

        # Disable maximize button
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        # Set the current page to `loginScanFace_page`
        self.stackWidget.setCurrentWidget(self.loginScanFace_page)
        self.exitButton.clicked.connect(self.exit_to_login)

        # Initialize camera
        self.timer = QTimer()
        self.camera = cv2.VideoCapture(0)  # Open the default camera (ID 0)

        if not self.camera.isOpened():
            print("Error: Camera not found!")
            return

        # Load known faces and encodings from the database
        self.known_face_encodings, self.known_face_names = self.load_faces_from_database()

        # Initialize duration tracking
        self.elapsed_time = 0  # Time in seconds
        self.max_time = 7 * 60  # 7 minutes in seconds

        # Connect the timer to update the camera feed
        self.timer.timeout.connect(self.login_scanFace)
        self.timer.start(30) 

    def exit_to_login(self):
        """Handle the exit button click to go back to the login screen."""
        # Close the current window
        self.close()

        # Create an instance of the LoginScreen and show it
        self.login_window = LoginScreen()  # Instantiate LoginScreen, which has the button logic
        self.login_window.show()


    def load_faces_from_database(self):
        """Load face encodings and names from the database."""
        try:
            conn = psycopg2.connect(
                host='localhost',
                dbname='Facelog',
                user='postgres',
                password='password',
                port=5433
            )
            cursor = conn.cursor()
            cursor.execute("SELECT firstname, face_data, face_image FROM users")
            results = cursor.fetchall()
            cursor.close()
            conn.close()

            known_face_encodings = []
            known_face_names = []

            # Iterate through the results and unpack the columns correctly
            for result in results:
                firstname = result[0]  # Extract firstname
                face_data = result[1]   # Extract face_data (encoding)
                face_image = result[2]  # Extract face_image (image)

                print(f"Name: {firstname}, Data Type: {type(face_data)}, Length: {len(face_data)}")

                # Debugging: Show the first 20 bytes in a more human-readable format
                face_data_bytes = bytes(face_data)
                print(f"First 20 bytes of face data: {face_data_bytes[:20]}")

                # Debugging: Attempt to deserialize the face_data
                face_data_buffer = io.BytesIO(face_data_bytes)
                try:
                    encoding = np.load(face_data_buffer, allow_pickle=True)
                    print(f"Loaded encoding of length: {len(encoding)}")
                    known_face_encodings.append(encoding)
                    known_face_names.append(firstname)
                except Exception as e:
                    print(f"Error during deserialization: {e}")

            return known_face_encodings, known_face_names

        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load faces from database: {e}")
            return [], []


    def login_scanFace(self):
        """Recognize the user's face and display their data if recognized."""
        # Check if the maximum time has been reached
        self.elapsed_time += 0.5  # Increment time by 0.5 seconds (500ms timer)
        if self.elapsed_time >= self.max_time:
            QMessageBox.warning(self, "Timeout", "Face scanning time has expired.")
            self.timer.stop()
            self.camera.release()
            return

        ret, frame = self.camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            return

        # Flip the frame to avoid a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Draw bounding boxes around detected faces
        for (top, right, bottom, left) in face_locations:
            # Draw rectangle around each detected face
            color = (0, 255, 0)  # Green color
            thickness = 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

        # Update the QLabel with the current frame
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_frame = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_frame)

        # Set the updated pixmap in QLabel
        self.cameraFeedLabel.setPixmap(pixmap)
        self.cameraFeedLabel.setAlignment(Qt.AlignCenter)  # Center the feed in QLabel
        self.cameraFeedLabel.repaint()  # Force QLabel to refresh

        face_recognized = False  # Flag to check if a face is recognized

        for face_encoding in face_encodings:
            # Compute distances between the current face and known encodings
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            print(f"Face distances: {face_distances}")  # Debugging distances

            # Find the best match index
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] <= 0.4:  # Match within tolerance
                name = self.known_face_names[best_match_index]
                print(f"Recognized: {name} with distance {face_distances[best_match_index]}")

                # Retrieve and display user data based on name
                self.display_user_data(name)

                # Stop camera and timer
                self.timer.stop()
                self.camera.release()
                face_recognized = True
                return

        # If face is not recognized after a certain duration, navigate to the registration page
        if not face_recognized and self.elapsed_time >= 15:  # Allow 15 seconds for recognition
            reply = QMessageBox.warning(self, "Unrecognized Face", "Face not recognized. Redirecting to registration.",
                                        QMessageBox.Ok)

            if reply == QMessageBox.Ok:
                # Stop the camera
                self.timer.stop()
                self.camera.release()

                # Close the current UserScreen and open RegistrationScreen
                self.open_registration()
    
    def display_user_data(self, name):
        """Retrieve and display user data based on the recognized name."""
        try:
            conn = psycopg2.connect(
                host='localhost',
                dbname='Facelog',
                user='postgres',
                password='password',
                port=5433
            )
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE firstname = %s", (name,))
            user_data = cursor.fetchone()
            cursor.close()
            conn.close()

            if user_data:
                # Unpack all the values from the database
                (
                    id_num, firstname, lastname, email, phone_number, birthdate, age, gender,
                    work_status, civil_status, mother_firstname, mother_lastname,
                    father_firstname, father_lastname, guardian_firstname, guardian_lastname,
                    street_temp, barangay_temp, city_temp, state_temp, zip_code_temp,
                    street_perm, barangay_perm, city_perm, state_perm, zip_code_perm, face_data, face_image
                ) = user_data

                # Display user details in labels
                self.userIDLabel.setText(str(id_num))
                self.nameLabel.setText(f"{firstname} {lastname}")
                self.emailLabel.setText(email)
                self.phoneLabel.setText(phone_number)
                self.birthdateLabel.setText(str(birthdate))
                self.ageLabel.setText(str(age))
                self.genderLabel.setText(gender)
                self.workStatusLabel.setText(work_status)
                self.civilStatusLabel.setText(civil_status)

                self.fatherLabel.setText(f"{father_firstname} {father_lastname}")
                self.motherLabel.setText(f"{mother_firstname} {mother_lastname}")
                self.guardianLabel.setText(f"{guardian_firstname} {guardian_lastname}")
                # Display addresses (Temporary and Permanent)
                self.tempAddress.setText(f"{street_temp}, {barangay_temp}, {city_temp}, {state_temp}")
                self.tempCode.setText(zip_code_temp)

                self.permAddress.setText(f"{street_perm}, {barangay_perm}, {city_perm}, {state_perm}")
                self.permCode.setText(zip_code_perm)

                # Process and display the user's face image
                if face_image:
                    try:
                        # Convert the face data from the database to a usable image
                        face_image = bytes(face_image)  # Convert memoryview to bytes
                        image = QImage.fromData(face_image)

                        if not image.isNull():
                            pixmap = QPixmap.fromImage(image)

                            # Scale the image to exactly fit the QLabel size
                            scaled_pixmap = pixmap.scaled(
                                self.reviewFaceImage.width(),
                                self.reviewFaceImage.height(),
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation
            )

                            # Set the scaled pixmap to the QLabel
                            self.faceDisplay.setPixmap(scaled_pixmap)
                            self.faceDisplay.setAlignment(Qt.AlignCenter)  # Center the image inside the label
                        else:
                            print("Face image data is invalid.")
                            self.faceDisplay.clear()
                    except Exception as e:
                        print(f"Failed to process face data: {e}")
                        self.faceDisplay.clear()


                # Navigate to the displayProfile_page in stackWidget
                self.stackWidget.setCurrentWidget(self.displayProfile_page)

            else:
                QMessageBox.warning(self, "User Not Found", "No user data found for the given name.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve user data: {e}")



    def open_registration(self):
        """Switch to the registration screen."""
        self.registration_screen = RegistrationScreen()
        self.registration_screen.show()
        self.close()

    def closeEvent(self, event):
        """Release resources when the UserScreen is closed."""
        self.timer.stop()
        self.camera.release()
        cv2.destroyAllWindows()
        super(UserScreen, self).closeEvent(event)


class RegistrationScreen(QMainWindow):
    def __init__(self):
        super(RegistrationScreen, self).__init__()
        loadUi(r"C:\Users\angelica\Downloads\FaceLog\user.ui", self)

        # Disable maximize button
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        # Set the initial page to registration1_page
        self.stackWidget.setCurrentWidget(self.registration1_page)

        # Connect buttons
        self.nextButton1.clicked.connect(self.check_phone_and_email_uniqueness)
        self.nextButton2.clicked.connect(self.start_face_registration)
        self.backButton.clicked.connect(self.go_to_registration1_page)
        self.submitButton.clicked.connect(self.display_review_page)
        self.editButton.clicked.connect(self.enable_edit)
        self.registerButton.clicked.connect(self.register_user)

        # Connect birthdate input to age calculation
        self.birthdateInput.dateChanged.connect(self.update_age)

        # Initialize camera and timer, but don't start it yet
        self.timer = QTimer()
        self.camera = None

     

    def create_database_connection(self):
        """Create and return a connection to the PostgreSQL database."""
        return psycopg2.connect(
            host='localhost',
            dbname='Facelog',
            user='postgres',
            password='password',
            port=5433
        )

    def update_age(self):
        """Calculate and update the age based on the selected birthdate."""
        birthdate = self.birthdateInput.date()
        current_date = QDate.currentDate()

        # Calculate the age
        age = current_date.year() - birthdate.year()

        # Adjust age if the birthday has not occurred yet this year
        if (current_date.month() < birthdate.month()) or \
                (current_date.month() == birthdate.month() and current_date.day() < birthdate.day()):
            age -= 1

        # Update the age in the QLineEdit
        self.ageInput.setText(str(age))

    
    def go_to_registration1_page(self):
        """Switch back to the registration1_page."""
        self.stackWidget.setCurrentWidget(self.registration1_page)

    def go_to_registration2_page(self):
        """Switch to the registration2_page to collect guardian and address information."""
        # Validate fields in registration1_page before moving to registration2_page
        user_data = {
            'fname': self.fnameInput.text().strip(),
            'lname': self.lnameInput.text().strip(),
            'email': self.emailInput.text().strip(),
            'phone': self.phoneInput.text().strip(),
            'birthdate': self.birthdateInput.date().toString("yyyy-MM-dd"),
            'gender': self.genderComboBox.currentText().strip(),
            'work_status': self.workStatusComboBox.currentText().strip(),
            'civil_status': self.civilStatusComboBox.currentText().strip(),
            'mother_fname': self.motherFnameInput.text().strip(),
            'mother_lname': self.motherLnameInput.text().strip(),
            'father_fname': self.fatherFnameInput.text().strip(),
            'father_lname': self.fatherLnameInput.text().strip()
        }

        # Check for missing fields in registration1_page
        errors = []
        for key, value in user_data.items():
            if not value:
                errors.append(f"{key.replace('_', ' ').title()} cannot be empty.")

        # Display errors if any exist
        if errors:
            QMessageBox.warning(self, "Invalid or Incomplete Data", "\n".join(errors))
            return

        # If all validations pass, move to registration2_page
        self.stackWidget.setCurrentWidget(self.registration2_page)
    
    def validate_registration2_page(self):
        """Validate the fields of registration2_page before proceeding."""
        guardian_data = {
            'guardian_fname': self.guardianFnameInput.text().strip(),
            'guardian_lname': self.guardianLnameInput.text().strip(),
            'street_temp': self.streetInputTemp.text().strip(),
            'barangay_temp': self.barangayInputTemp.text().strip(),
            'city_temp': self.cityInputTemp.text().strip(),
            'state_temp': self.stateInputTemp.text().strip(),
            'zip_code_temp': self.zipCodeInputTemp.text().strip(),
            'street_perm': self.streetInputPerm.text().strip(),
            'barangay_perm': self.barangayInputPerm.text().strip(),
            'city_perm': self.cityInputPerm.text().strip(),
            'state_perm': self.stateInputPerm.text().strip(),
            'zip_code_perm': self.zipCodeInputPerm.text().strip()
        }

        # Check for missing fields
        errors = []
        for key, value in guardian_data.items():
            if not value:
                errors.append(f"{key.replace('_', ' ').title()} cannot be empty.")

        if errors:
            QMessageBox.warning(self, "Incomplete Data", "\n".join(errors))
            return False  # Validation failed

        return True  # Validation successful

    def check_phone_and_email_uniqueness(self):
        """Check if the phone number and email are unique and valid before proceeding to the next page."""
        try:
            phone_number = self.phoneInput.text().strip()
            email = self.emailInput.text().strip()

            # Validate phone number length
            if not phone_number.isdigit() or len(phone_number) != 11:
                QMessageBox.warning(self, "Invalid Phone Number", "Phone number must be 11 digits.")
                return

            # Validate email format
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_regex, email):
                QMessageBox.warning(self, "Invalid Email", "Please enter a valid email address.")
                return

            # Connect to the database
            conn = self.create_database_connection()
            cursor = conn.cursor()

            # Check if the phone number already exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE phone_number = %s", (phone_number,))
            phone_count = cursor.fetchone()[0]

            if phone_count > 0:
                QMessageBox.warning(self, "Duplicate Entry", "The phone number is already registered.")
                cursor.close()
                conn.close()
                return

            # Check if the email already exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            email_count = cursor.fetchone()[0]

            if email_count > 0:
                QMessageBox.warning(self, "Duplicate Entry", "The email is already registered.")
                cursor.close()
                conn.close()
                return

            # If both are valid and unique, proceed to the next page
            cursor.close()
            conn.close()
            self.go_to_registration2_page()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to validate phone number or email: {e}")


    def start_face_registration(self):
        # Validate fields in registration2_page
        if not self.validate_registration2_page():
            return  # Exit if validation fails
        
        """Start the camera feed once the user clicks the 'Next' button on registration2_page."""
        # Initialize camera and timer
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Camera Error", "Camera not found!")
            return

        self.timer.timeout.connect(self.register_face)
        self.timer.start(30)  # Refresh rate of 30ms to capture faces

        # Switch to the face registration page (registerFace_page)
        self.stackWidget.setCurrentWidget(self.registerFace_page)

    def register_face(self):
        """Capture and display the face during registration (on registerFace_page)."""
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            return

        # Flip the frame for a mirrored display
        frame = cv2.flip(frame, 1)

        # Detect face using Haar Cascade or similar (assuming one is implemented)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            center_x, center_y = x + w // 2, y + h // 2
            start_x, start_y = center_x - int(w * 0.6), center_y - int(h * 0.6)
            end_x, end_y = center_x + int(w * 0.6), center_y + int(h * 0.6)

            # Adjust rectangle to stay within frame boundaries
            start_x, start_y = max(start_x, 0), max(start_y, 0)
            end_x, end_y = min(end_x, frame.shape[1]), min(end_y, frame.shape[0])

            # Draw a centered rectangle around the face
            color = (0, 255, 0)  # Green color
            thickness = 2  # Thickness of the rectangle
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, thickness)

        # Convert the frame to RGB for PyQt display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to QImage for QLabel
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        qimg_resized = qimg.scaled(365, 351, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.registerFace.setPixmap(QPixmap.fromImage(qimg_resized))

    def display_review_page(self):
        """Display the user data and captured face image on reviewRegistration_page."""
        try:
            # Capture the final face image
            ret, frame = self.camera.read()
            if not ret:
                QMessageBox.critical(self, "Capture Error", "Failed to capture face image!")
                return

            # Stop the camera and release resources
            self.stop_camera()

            # Convert BGR (OpenCV format) to RGB (face_recognition format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face locations and compute encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            if not face_locations:
                QMessageBox.critical(self, "No Face Detected", "Please ensure your face is visible!")
                self.stop_camera()
                return


            # Get the encoding of the first face (assuming one face in the frame)
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            # Serialize face encoding for database storage
            face_data_buffer = io.BytesIO()
            np.save(face_data_buffer, face_encoding, allow_pickle=True)  # Save numpy array to buffer
            self.face_data = face_data_buffer.getvalue()  # Get the binary content

            # Convert the frame to a buffer (for image storage)
            _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as a JPEG
            self.face_image = buffer.tobytes()  # Store the image as binary data

            # Collect all data from registration1_page and registration2_page
            self.user_data = {
                'fname': self.fnameInput.text(),
                'lname': self.lnameInput.text(),
                'email': self.emailInput.text(),
                'phone': self.phoneInput.text(),
                'birthdate': self.birthdateInput.date().toString("yyyy-MM-dd"),
                'age': self.ageInput.text(),
                'gender': self.genderComboBox.currentText(),
                'work_status': self.workStatusComboBox.currentText(),
                'civil_status': self.civilStatusComboBox.currentText(),
                'mother_fname': self.motherFnameInput.text(),
                'mother_lname': self.motherLnameInput.text(),
                'father_fname': self.fatherFnameInput.text(),
                'father_lname': self.fatherLnameInput.text(),
                'guardian_fname': self.guardianFnameInput.text(),
                'guardian_lname': self.guardianLnameInput.text(),
                'street_temp': self.streetInputTemp.text(),
                'barangay_temp': self.barangayInputTemp.text(),
                'city_temp': self.cityInputTemp.text(),
                'state_temp': self.stateInputTemp.text(),
                'zip_code_temp': self.zipCodeInputTemp.text(),
                'street_perm': self.streetInputPerm.text(),
                'barangay_perm': self.barangayInputPerm.text(),
                'city_perm': self.cityInputPerm.text(),
                'state_perm': self.stateInputPerm.text(),
                'zip_code_perm': self.zipCodeInputPerm.text()
            }

            # Concatenate addresses
            temp_address = f"{self.user_data['street_temp']}, {self.user_data['barangay_temp']}, {self.user_data['city_temp']}, {self.user_data['state_temp']}"
            perm_address = f"{self.user_data['street_perm']}, {self.user_data['barangay_perm']}, {self.user_data['city_perm']}, {self.user_data['state_perm']}"

            # Populate fields on the reviewRegistration_page
            self.reviewFname.setText(self.user_data['fname'])
            self.reviewLname.setText(self.user_data['lname'])
            self.reviewEmail.setText(self.user_data['email'])
            self.reviewPhone.setText(self.user_data['phone'])
            self.reviewBirthdate.setText(self.user_data['birthdate'])
            self.reviewAge.setText(self.user_data['age'])
            self.reviewGender.setText(self.user_data['gender'])
            self.reviewWorkStatus.setText(self.user_data['work_status'])
            self.reviewCivilStatus.setText(self.user_data['civil_status'])
            self.reviewMotherFname.setText(self.user_data['mother_fname'])
            self.reviewMotherLname.setText(self.user_data['mother_lname'])
            self.reviewFatherFname.setText(self.user_data['father_fname'])
            self.reviewFatherLname.setText(self.user_data['father_lname'])
            self.reviewGuardianFname.setText(self.user_data['guardian_fname'])
            self.reviewGuardianLname.setText(self.user_data['guardian_lname'])
            self.reviewTempAddress.setText(temp_address)
            self.reviewPermAddress.setText(perm_address)
            self.reviewPermZipcode.setText(self.user_data['zip_code_perm'])
            self.reviewTempZipcode.setText(self.user_data['zip_code_temp'])

            # Convert the captured frame to a buffer for QImage
            _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as a JPEG

            # Convert the buffer to QImage
            image = QImage.fromData(buffer.tobytes())  # Create QImage from encoded data

            if image.isNull():
                QMessageBox.critical(self, "Image Error", "Failed to process captured face image!")
                return

            # Resize the image to fit the QLabel dimensions while keeping the aspect ratio
            qpixmap = QPixmap.fromImage(image)
            scaled_pixmap = qpixmap.scaled(
                self.reviewFaceImage.width(),
                self.reviewFaceImage.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Set the QLabel to display the scaled image
            self.reviewFaceImage.setPixmap(scaled_pixmap)

            # Switch to the reviewRegistration_page
            self.stackWidget.setCurrentWidget(self.reviewRegistration_page)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display review page: {e}")

    def stop_camera(self):
        """Stop the camera and release resources."""
        if self.timer.isActive():
            self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        self.registerFace.clear()

    def enable_edit(self):
        """Enable editing of all fields on the reviewRegistration_page."""
        self.stackWidget.setCurrentWidget(self.registration1_page)  # Allow user to navigate back to edit

    def register_user(self):
        """Save all user data and face image to the PostgreSQL database."""
        try:
            conn = self.create_database_connection()
            cursor = conn.cursor()

            # Check if the phone number already exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE phone_number = %s", (self.user_data['phone'],))
            count = cursor.fetchone()[0]

            if count > 0:
                QMessageBox.warning(self, "Duplicate Entry", "The phone number is already registered.")
                cursor.close()
                conn.close()
                return

            cursor.execute(
                """INSERT INTO users (
                    firstname, lastname, email, phone_number, birthdate, age, gender, 
                    work_status, civil_status, mother_firstname, mother_lastname, 
                    father_firstname, father_lastname, guardian_firstname, guardian_lastname,
                    street_temp, barangay_temp, city_temp, state_temp, zip_code_temp,
                    street_perm, barangay_perm, city_perm, state_perm, zip_code_perm, face_data, face_image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                tuple(self.user_data.values()) + (psycopg2.Binary(self.face_data),) +  # Face encoding as binary
            (psycopg2.Binary(self.face_image),)  # Face image as binary (JPEG) # Store as binary data
            )

            conn.commit()
            cursor.close()
            conn.close()

            QMessageBox.information(self, "Success", "User data and face image saved successfully!")

        # Close the Registration Screen and open Login Screen
            self.close()

            # Create and show the Login Screen
            self.login_screen = LoginScreen()  # Replace with your actual login UI class
            self.login_screen.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {e}") 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginScreen()  # Make sure the LoginScreen is shown first
    window.show()  # Show the LoginScreen
    sys.exit(app.exec_())  # Run the event loop

