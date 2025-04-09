import psycopg2
import torch
from config.db_config import DB_HOST, PORT, DB_DATABASE, DB_USER, DB_PASSWORD
from datetime import datetime, timedelta
import uuid
import src.utils.convert_type as convert_type
import numpy as np
from pgvector.psycopg2 import register_vector
def get_connection():
    conn=psycopg2.connect(
        host=DB_HOST,
        port=PORT,
        dbname=DB_DATABASE,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

def get_all_classes_in_next_2_hours():
    conn=get_connection()
    cur=conn.cursor()
    datetime_now=datetime.now()
    two_hours_later=datetime_now+ timedelta(hours=2)
    query="""
       SELECT s.schedule_id, s.class_id, s.classroom_id, cam.camera_URL, cam.socket_path 
        FROM schedules s
        JOIN classes c ON s.class_id = c.class_id
		JOIN cameras cam ON cam.classroom_id=cam.classroom_id
        WHERE s.start_time BETWEEN %s AND %s AND cam.camera_type='recognition';
        """
    cur.execute(query,(datetime_now,two_hours_later))
    rows=cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_class_in_next_2_hours(class_id):
    conn=get_connection()
    cur=conn.cursor()
    datetime_now=datetime.now()
    two_hours_later=datetime_now+ timedelta(hours=2)
    query="""
       SELECT s.class_id, s.classroom_id, cam.camera_ip, cam.socket_path
        FROM schedules s
        JOIN classes c ON s.class_id = c.class_id
		JOIN cameras cam ON cam.classroom_id=cam.classroom_id
        WHERE s.start_time BETWEEN %s AND %s AND s.classroom_id=%s AND cam.camera_type='recognition';
        """
    cur.execute(query,(datetime_now,two_hours_later,class_id))
    rows=cur.fetchall()
    cur.close()
    conn.close()
    return rows
def get_students_embedding_for_classes(class_ids):
    conn=get_connection()
    cur=conn.cursor()
    class_ids = [str(uuid.UUID(cid)) for cid in class_ids]  # Chuyển class_ids thành chuỗi UUID
    print(class_ids)    
    query="""
        SELECT 
        s.student_id, u.first_name || ' ' || u.last_name || ' - ' || s.student_code as full_name,c.class_id, s.face_embedding
        FROM classes c 
        JOIN class_students cs ON c.class_id=cs.class_id
        JOIN students s ON s.student_id=cs.student_id
        JOIN users u ON cs.student_id=u.user_id
        WHERE c.class_id = ANY(ARRAY[%s]::UUID[])
        """
    # if isinstance(class_ids, list):
    # class_ids = tuple(class_ids)
    cur.execute(query,(class_ids,))
    rows=cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_students_embedding_for_one_class(class_id):
    conn=get_connection()
    cur=conn.cursor()
    query="""
        SELECT 
        s.student_id, u.first_name || ' ' || u.last_name || ' - ' || s.student_code as full_name, s.face_embedding
        FROM classes c 
        JOIN class_students cs ON c.class_id=cs.class_id
        JOIN students s ON s.student_id=cs.student_id
        JOIN users u ON cs.student_id=u.user_id
        WHERE c.class_id = %s
        """
    # if isinstance(class_ids, list):
    # class_ids = tuple(class_ids)
    cur.execute(query,(class_id,))
    rows=cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_camera(room_id):
    conn=get_connection()
    cur=conn.cursor()
    query="""
        SELECT cam.camera_id,camera_ip
        FROM classrooms c
        JOIN cameras cam ON c.classroom_id=cam.classroom_id
        WHERE c.classroom_id=%s AND cam.camera_type='recognition'
        LIMIT 1;    
        """
    cur.execute(query,(room_id,))
    row=cur.fetchone()
    cur.close() 
    conn.close()
    return row

def update_embedding(student_id,embedding):
    conn=get_connection()
    cur=conn.cursor()
    register_vector(conn)

    query="UPDATE students SET face_embedding = %s WHERE student_id = %s"
    cur.execute(query, (embedding.tolist(), student_id))
    conn.commit()
    cur.close()
    conn.close()
#        s.student_id, u.first_name || ' ' || u.last_name || ' - ' || s.student_code as full_name,c.class_id, s.face_embedding

def get_pytorch_embedding(class_id):
    """Fetch student embeddings from PostgreSQL and convert to PyTorch tensors."""
    print(f"📌 Fetching embeddings for class: {class_id}")

    # 🔹 Fetch data from PostgreSQL
    rows = get_students_embedding_for_one_class(class_id)
    
    if not rows:
        print("❌ No students found for this class.")
        return torch.empty(0), []

    embeddings = []
    student_ids = []

    for row in rows:
        student_id, full_name, embedding = row
        student_ids.append((student_id, full_name))
        
        # 🔹 Convert string embedding to NumPy array
        embedding_array = convert_type.string_to_np(embedding)
        
        # 🔹 Ensure it's a PyTorch tensor
        embedding_tensor = torch.tensor(embedding_array, dtype=torch.float32)
        embeddings.append(embedding_tensor)

    # 🔹 Convert list of tensors into a single stacked tensor
    targets = torch.stack(embeddings)

    print(f"✅ Loaded {len(student_ids)} embeddings for class {class_id}")
    
    return targets, student_ids


def attencace_student(schedule_id,student_id,image_path):
    datetime_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Định dạng chuẩn TIMESTAMP
    print('datetime_now',datetime_now)
    query="""
            INSERT INTO attendance (schedule_id, student_id, status, attendance_time, evidence_image_url)
            VALUES (%s, %s, 
                (CASE 
                    WHEN %s > (SELECT start_time FROM schedules WHERE schedule_id = %s) THEN 'late'
                    ELSE 'present'
                END), 
                %s, %s)
            ON CONFLICT (schedule_id, student_id) 
            DO UPDATE SET 
                status = EXCLUDED.status,
                attendance_time = EXCLUDED.attendance_time,
                evidence_image_url = EXCLUDED.evidence_image_url;
        """
    try:
        conn=get_connection()
        cur=conn.cursor()
        cur.execute(query,(schedule_id,student_id,datetime_now,schedule_id,datetime_now,image_path,))
        conn.commit()
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật điểm danh: {e}")
    finally:
        # Đóng kết nối
        if cur:
            cur.close()
        if conn:
            conn.close()
    
    
def init_attendace(schedule_id):
    print(f"📋 Đang khởi tạo điểm danh cho schedule_id = {schedule_id}")
    query="""
        INSERT INTO attendance (schedule_id, student_id, status, attendance_time)
        SELECT s.schedule_id, cs.student_id, 'absent', NULL
        FROM class_students cs
        JOIN schedules s ON cs.class_id = s.class_id
        WHERE s.schedule_id = %s
        ON CONFLICT (schedule_id, student_id) DO NOTHING;
    """
    try:
        conn=get_connection()
        cur=conn.cursor()

        cur.execute(query,(schedule_id,))
        conn.commit()
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật điểm danh: {e}")
    finally:
        # Đóng kết nối
        if cur:
            cur.close()
        if conn:
            conn.close()