import tkinter as tk
from tkinter import messagebox, ttk, PhotoImage
from PIL import Image, ImageTk
import cv2
from tkcalendar import DateEntry
import mysql.connector
import os
import face_recognition
import datetime
from ultralytics import YOLO
import pandas as pd
import numpy as np
import time
import re


model = YOLO(r"C:\Users\hindi\PycharmProjects\PythonProject\face_detection\yolov8n.pt")

# ----------------- Entry Focus Functions -------------------

def on_click_entry(event):
    if e1.get() == "Username":
        e1.delete(0, "end")
        e1.config(fg="black")

def on_focusout(event):
    if e1.get() == "":
        e1.insert(0, "Username")
        e1.config(fg="gray")

def on_click_password(event):
    if e2.get() == "Password":
        e2.delete(0, "end")
        e2.config(fg="black", show="*")

def on_password(event):
    if e2.get() == "":
        e2.insert(0, "Password")
        e2.config(fg="gray", show="")

# ----------------- LOGIN FUNCTION -------------------

def login():
    def validate_login():
        username = e1.get().strip()
        password = e2.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Both fields are required")
            return

        try:
            conn = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="",
                database="students_db"
            )
            cur = conn.cursor()
            cur.execute("SELECT * FROM admin WHERE username=%s AND password=%s", (username, password))
            result = cur.fetchone()
            conn.close()

            if result:
                messagebox.showinfo("Login Success", f"Welcome {username}!")
                show_main_window(username)  # Show the main window content instead of opening a new window
            else:
                messagebox.showerror("Login Failed", "Invalid username or password")

        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", str(err))

    validate_login()

# ----------------- SIGNUP FUNCTION -------------------

def signup():
    t = tk.Toplevel(root)
    t.geometry("300x550")
    t.title("Admin Registration")

    t.transient(root)

    img2 = Image.open("imgs\\i3.jpg")
    img2 = img2.resize((300, 200))
    img2 = ImageTk.PhotoImage(img2)
    l_img2 = tk.Label(t, image=img2)
    l_img2.image = img2
    l_img2.pack()

    tk.Label(t, text="Username").pack()
    tu1 = tk.Entry(t)
    tu1.pack()

    tk.Label(t, text="Name").pack()
    te1 = tk.Entry(t)
    te1.pack()

    tk.Label(t, text="Department").pack()
    te2 = tk.Entry(t)
    te2.pack()

    tk.Label(t, text="DOB").pack()
    dob_entry = DateEntry(t)
    dob_entry.pack()

    tk.Label(t, text="Email").pack()
    te4 = tk.Entry(t)
    te4.pack()

    tk.Label(t, text="Password").pack()
    te5 = tk.Entry(t, show="*")
    te5.pack()

    def click_photo():
        username = tu1.get().strip()
        if not username:
            messagebox.showerror("Missing Username", "Please enter a username before clicking a photo.")
            return

        os.makedirs("admin_faces", exist_ok=True)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Failed to access the webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            '''faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)'''

            cv2.imshow("Face Detection - Press 's' to Save, 'q' to Quit", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                photo_path = f"admin_faces/{username}.jpg"
                cv2.imwrite(photo_path, frame)
                t.photo_path = photo_path
                messagebox.showinfo("Success", f"Photo saved as {photo_path}")
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    tk.Button(t, text="Click Photo", command=click_photo).pack(pady=5)

    def submit_signup():
        username = tu1.get().strip()
        name = te1.get().strip()
        dept = te2.get().strip()
        dob = dob_entry.get_date()
        email = te4.get().strip()
        password = te5.get().strip()
        photo_path = getattr(t, "photo_path", None)

        if not all([username, name, dept, dob, email, password]):
            messagebox.showerror("Error", "All fields are required")
            return
        if not photo_path:
            messagebox.showerror("Error", "Please click a photo before submitting")
            return

        if len(username) < 4 or not username.isalnum():
            messagebox.showerror("Error", "Username must be at least 4 characters and alphanumeric")
            return

        if not name.replace(" ", "").isalpha():
            messagebox.showerror("Error", "Name should contain only letters")
            return

        if not dept.replace(" ", "").isalpha():
            messagebox.showerror("Error", "Department should contain only letters")
            return

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            messagebox.showerror("Error", "Invalid email format")
            return

        if len(password) < 6:
            messagebox.showerror("Error", "Password must be at least 6 characters long")
            return

        try:
            conn = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="",
                database="students_db"
            )
            cur = conn.cursor()
            cur.execute("SELECT * FROM admin WHERE username = %s", (username,))
            if cur.fetchone():
                messagebox.showerror("Username Taken", "This username is already registered. Please choose another.")
                conn.close()
                return

            cur.execute(""" 
                INSERT INTO admin (username, name, department, dob, email, password, photo)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (username, name, dept, dob, email, password, photo_path))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "Admin registered successfully")
            t.destroy()

        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", str(err))

    tk.Button(t, text="Submit", command=submit_signup).pack(pady=10)

# ----------------- FACE LOGIN FUNCTION -------------------

def face_login():
    known_encodings = []
    known_usernames = []
    face_dir = "admin_faces"

    os.makedirs(face_dir, exist_ok=True)

    for filename in os.listdir(face_dir):
        if filename.endswith(".jpg"):
            username = os.path.splitext(filename)[0]
            img_path = os.path.join(face_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_usernames.append(username)

    if not known_encodings:
        messagebox.showerror("No Faces", "No registered admin faces found.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Failed to access the webcam.")
        return

    authenticated_user = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            best_match_index = None
            if face_distances.size > 0:
                best_match_index = face_distances.argmin()

            if best_match_index is not None and matches[best_match_index]:
                authenticated_user = known_usernames[best_match_index]
                break

        cv2.imshow("Face Login - Press 'q' to quit", frame)

        if authenticated_user:
            messagebox.showinfo("Login Success", f"Welcome back, {authenticated_user}!")
            show_main_window(authenticated_user)  # Show the main window content
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    if not authenticated_user:
        messagebox.showerror("Login Failed", "Face not recognized.")

    cap.release()
    cv2.destroyAllWindows()

# ----------------- MAIN WINDOW FUNCTION -------------------

def show_main_window(username=None):
    for widget in root.winfo_children():
        widget.destroy()



    if username:
        photo_path = os.path.join("admin_faces", f"{username}.jpg")
        if os.path.exists(photo_path):
            img = Image.open(photo_path)
            img = img.resize((50, 50))
            img_tk = ImageTk.PhotoImage(img)

            img_label = tk.Label(root, image=img_tk)
            img_label.image = img_tk  # Prevent garbage collection
            img_label.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

            tk.Label(root, text=f"Hi {username} !", font=("Arial",18)).pack(pady=10)


    lbl_main = tk.Label(root, text="Welcome to the Facial Attendance System!", font=('Arial', 16))
    lbl_main.pack(pady=20)

    tk.Label(root, text='With Dress Colour and Tie Compliance', font=('Arial', 14),fg='blue').pack(pady=10)

    def start_atten(event):

        # === Shirt Detection Using HSV ===
        def check_uniform_color(frame, face_location):
            top, right, bottom, left = face_location
            face_height = bottom - top
            torso_top = bottom
            torso_bottom = bottom + face_height

            h, w, _ = frame.shape
            torso_top = max(0, torso_top)
            torso_bottom = min(h, torso_bottom)
            left = max(0, left)
            right = min(w, right)

            torso = frame[torso_top:torso_bottom, left:right]
            if torso.size == 0:
                return False

            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            shirt_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            shirt_ratio = cv2.countNonZero(shirt_mask) / (torso.size / 3)

            return shirt_ratio > 0.25

        # === Tie Detection Using YOLO ===
        def detect_tie_yolo(model, image_path):
            results = model(image_path, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.2 and model.names[cls_id] == "tie":
                    return True
            return False

        face_dir = "student_faces"
        os.makedirs(face_dir, exist_ok=True)

        temp_dir = "temp_detection"
        os.makedirs(temp_dir, exist_ok=True)

        known_encodings = []
        known_names = []

        # Load registered student faces
        for filename in os.listdir(face_dir):
            if filename.endswith(".jpg"):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(face_dir, filename)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)

        if not known_encodings:
            messagebox.showerror("No Students", "No student faces registered.")
            return

        csv_file = "attendance.csv"
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
            existing_df = pd.DataFrame(columns=["Name", "Date", "Time", "Dress Code"])
            existing_df.to_csv(csv_file, index=False)
        else:
            existing_df = pd.read_csv(csv_file)
            required_cols = {"Name", "Date", "Time", "Dress Code"}
            if not required_cols.issubset(set(existing_df.columns)):
                existing_df = pd.DataFrame(columns=["Name", "Date", "Time", "Dress Code"])
                existing_df.to_csv(csv_file, index=False)

        already_marked = set(existing_df[existing_df["Date"] == today]["Name"].values)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(0.5)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            window_name = "Attendance System - Press 'q' to Quit"
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                if len(face_distances) > 0:
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                        if name not in already_marked:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_filename = os.path.join(temp_dir, f"{name}_{timestamp}.jpg")
                            cv2.imwrite(img_filename, frame)

                            # === Detect dress code ===
                            has_tie = detect_tie_yolo(model, img_filename)
                            has_shirt = check_uniform_color(frame, face_location)

                            dress_code = "Pass" if has_shirt and has_tie else "Fail"

                            now = datetime.datetime.now()
                            new_row = {
                                "Name": name,
                                "Date": now.strftime("%Y-%m-%d"),
                                "Time": now.strftime("%H:%M:%S"),
                                "Dress Code": dress_code
                            }

                            existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
                            existing_df.to_csv(csv_file, index=False)

                            already_marked.add(name)

                            if dress_code == "Pass":
                                messagebox.showinfo("Attendance", f"Marked attendance for {name}")
                            else:
                                messagebox.showwarning("Dress Code", f"{name} is not wearing shirt or tie!")

                        # Draw rectangle and name
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            '''cv2.imshow("Attendance System - Press 'q' to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break'''


        cap.release()
        cv2.destroyAllWindows()

    tk.Label(root, text='Click Red Button to start taking Attendance', fg='green').pack(pady=5)

    def on_circle_click(event):
        # Shrink and darken temporarily
        canvas.itemconfig(circle, fill="blue")
        canvas.coords(circle, 15, 15, 185, 185)  # Smaller size
        root.after(10000, reset_circle)  # Restore after 100 ms

        # Call the attendance function
        start_atten(event)

    def reset_circle():
        canvas.itemconfig(circle, fill="red")
        canvas.coords(circle, 10, 10, 190, 190)

    canvas = tk.Canvas(root, width=200, height=200, highlightthickness=0)
    canvas.pack(pady=50)

    shadow = canvas.create_oval(15, 15, 195, 195, fill="gray", outline="")

    # Draw a red circle (oval) on the canvas
    circle = canvas.create_oval(10, 10, 190, 190, fill="red", outline="")

    # Bind the circle to the click event
    canvas.tag_bind(circle, "<Button-1>", on_circle_click)





    def add_student():
        add_win = tk.Toplevel(root)
        add_win.title("Add Student")
        add_win.geometry("500x500")
        add_win.config()

        add_win.transient(root)
        add_win.grab_set()

        add_stu_pho = Image.open("imgs\\add_photo.png")
        add_stu_pho = add_stu_pho.resize((500, 100))
        add_stu_pho = ImageTk.PhotoImage(add_stu_pho)
        add_stu_pho_label = tk.Label(add_win, image=add_stu_pho)
        add_stu_pho_label.image = add_stu_pho
        add_stu_pho_label.pack()


        tk.Label(add_win, text="Name").pack()
        name_entry = tk.Entry(add_win)
        name_entry.pack()

        tk.Label(add_win, text="Roll No").pack()
        roll_entry = tk.Entry(add_win)
        roll_entry.pack()

        tk.Label(add_win, text="Course").pack()
        course_entry = tk.Entry(add_win)
        course_entry.pack()



        photo_path_var = tk.StringVar()



        def capture_photo():
            name = name_entry.get().strip()
            roll_no = roll_entry.get().strip()
            course = course_entry.get().strip()

            if not name or not roll_no or not course:
                messagebox.showerror("Missing Info", "Please fill all fields before capturing photo.")
                return

            os.makedirs("student_faces", exist_ok=True)
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            while True:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("Capture Photo - Press 's' to Save", frame)

                key = cv2.waitKey(1)
                if key == ord('s') :
                    filename = f"{name}_{roll_no}.jpg"
                    path = os.path.join("student_faces", filename)
                    cv2.imwrite(path, frame)
                    photo_path_var.set(path)
                    messagebox.showinfo("Saved", f"Photo saved as {path}")
                    cap.release()
                    cv2.destroyAllWindows()
                    show_img()
                    photo_button.config(text="Repalce Photo")
                    return
                elif key == ord('q') :
                    break

            cap.release()
            cv2.destroyAllWindows()


        def show_img():
            path = photo_path_var.get()
            if os.path.exists(path):
                img = Image.open(path)
                img = img.resize((150, 150))
                img_tk = ImageTk.PhotoImage(img)
                img_label.configure(image=img_tk)
                img_label.image = img_tk


        def submit_student():
            name = name_entry.get().strip()
            roll_no = roll_entry.get().strip()
            course = course_entry.get().strip()
            photo_path = photo_path_var.get()

            if not roll_no.isalnum():
                messagebox.showerror("Validation Error", "Roll number must be alphanumeric.")
                return
            if len(name) < 2:
                messagebox.showerror("Validation Error", "Name must be at least 2 characters.")
                return

            if not all([name, roll_no, course, photo_path]):
                messagebox.showerror("Error", "Please fill all fields and capture a photo")
                return

            try:
                conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password="",
                    database="students_db"
                )
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO students (name, roll_no, course, photo_path) VALUES (%s, %s, %s, %s)",
                    (name, roll_no, course, photo_path)
                )
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", "Student added successfully")
                add_win.destroy()
            except mysql.connector.Error as err:
                messagebox.showerror("Database Error", str(err))

        photo_button=tk.Button(add_win, text="Capture Photo", command=capture_photo)
        photo_button.pack(pady=10)
        img_label = tk.Label(add_win)
        img_label.pack()
        tk.Button(add_win, text="Submit", command=submit_student).pack(pady=10)




    def show_attendance_window(csv_file="attendance.csv"):
        if not os.path.exists(csv_file):
            messagebox.showerror("Error", f"No attendance file found: {csv_file}")
            return

        df = pd.read_csv(csv_file)
        if df.empty or not {"Name", "Date", "Time", "Dress Code"}.issubset(df.columns):
            messagebox.showerror("Error", "Attendance file is empty or invalid.")
            return

        # Create new Tkinter window
        win = tk.Toplevel()
        win.title("Attendance Records")
        win.geometry("700x600")
        win.transient(root)

        # Dropdown to select date
        tk.Label(win, text="Select Date:").pack(pady=5)
        unique_dates = sorted(df["Date"].unique(), reverse=True)
        selected_date = tk.StringVar(value=unique_dates[0] if unique_dates else "")

        date_menu = tk.OptionMenu(win, selected_date, *unique_dates)
        date_menu.pack()

        tk.Label(win, text="Select Student:").pack(pady=5)
        unique_names = sorted(df["Name"].unique())
        selected_name = tk.StringVar(value=unique_names[0] if unique_names else "")
        name_menu = tk.OptionMenu(win, selected_name, *unique_names)
        name_menu.pack()

        df["Month"] = pd.to_datetime(df["Date"]).dt.strftime("%B %Y")
        unique_months = sorted(df["Month"].unique(), reverse=True)
        selected_month = tk.StringVar(value=unique_months[0] if unique_months else "")

        # Dropdown to select month
        tk.Label(win, text="Select Month:").pack(pady=5)
        month_menu = tk.OptionMenu(win, selected_month, *unique_months)
        month_menu.pack()

        # Monthly Attendance Summary
        def show_monthly_summary():
            name = selected_name.get()
            if not name:
                messagebox.showerror("Error", "No student selected.")
                return

            df_filtered = df[df["Name"] == name].copy()
            if df_filtered.empty:
                messagebox.showinfo("No Data", f"No attendance found for {name}.")
                return

            df_filtered["Month"] = pd.to_datetime(df_filtered["Date"]).dt.strftime("%B %Y")
            summary = df_filtered.groupby("Month").size()

            summary_text = f"Monthly Attendance Summary for {name}:\n\n"
            summary_text += "\n".join([f"{month}: {count} days" for month, count in summary.items()])
            messagebox.showinfo("Monthly Summary", summary_text)

        tk.Button(win, text="Show Monthly Attendance", command=show_monthly_summary).pack(pady=5)

        # Treeview to display records
        columns = ("Name", "Date", "Time", "Dress Code")
        tree = ttk.Treeview(win, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=150)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Populate Treeview with filtered data
        def update_treeview():
            for i in tree.get_children():
                tree.delete(i)
            selected = selected_date.get()
            filtered = df[df["Date"] == selected]
            for _, row in filtered.iterrows():
                tree.insert("", "end", values=tuple(row))

        update_treeview()

        # Refresh on date change
        def update_treeview():
            for i in tree.get_children():
                tree.delete(i)

            selected = selected_date.get()
            student = selected_name.get()
            month = selected_month.get()

            filtered = df[
                (df["Date"] == selected) &
                (df["Name"] == student) &
                (df["Month"] == month)
                ]

            for _, row in filtered.iterrows():
                tree.insert("", "end", values=tuple(row))

        selected_date.trace("w", lambda *args: update_treeview())
        selected_name.trace("w", lambda *args: update_treeview())
        selected_month.trace("w", lambda *args: update_treeview())

        # Optional: Close button
        tk.Button(win, text="Close", command=win.destroy).pack(pady=5)

    menu_bar = tk.Menu(root)
    menu_bar.add_cascade(label="Home")
    menu_bar.add_cascade(label="Attendance Records",command=show_attendance_window)
    menu_bar.add_cascade(label="Add Student",command=add_student)
    menu_bar.add_cascade(label="Exit", command=root.quit)
    root.config(menu=menu_bar)

# ----------------- UI SETUP -------------------

root = tk.Tk()
root.title("Welcome Back!")
root.geometry("600x500")
root.resizable(False, False)

img = Image.open("imgs/login_pic.jpg")
img = img.resize((300, 500))
img = ImageTk.PhotoImage(img)

limg = tk.Label(root, image=img)
limg.image = img
limg.place(x=0, y=0)

l1 = tk.Label(root, text="Welcome Back!", fg="blue", font=("Arial", 24))
l1.place(x=335, y=100)

e1 = tk.Entry(root, width=40)
e1.insert(0, "Username")
e1.config(fg="gray")
e1.bind('<FocusIn>', on_click_entry)
e1.bind('<FocusOut>', on_focusout)
e1.place(x=335, y=150)

e2 = tk.Entry(root, width=40)
e2.insert(0, "Password")
e2.config(fg="gray")
e2.bind('<FocusIn>', on_click_password)
e2.bind('<FocusOut>', on_password)
e2.place(x=335, y=180)

b1 = tk.Button(root, text="Login", command=login)
b1.place(x=430, y=210)

b2 = tk.Button(root, text="Click To Login Using Face Authentication", command=face_login)
b2.place(x=340, y=270)

l2 = tk.Label(root, text="New User? Sign up here")
l2.place(x=340, y=320)

b3 = tk.Button(root, text="Sign Up", command=signup)
b3.place(x=340, y=350)

root.mainloop()