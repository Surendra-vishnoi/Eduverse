from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, DateTime, Boolean, BigInteger, Integer, Text, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class User(Base):
    """User model - stores Google-authenticated users."""
    
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)  

    google_id = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    picture = Column(String(512), nullable=True)
    
    encrypted_access_token = Column(Text, nullable=True)
    encrypted_refresh_token = Column(Text, nullable=True)
    token_expiry = Column(DateTime, nullable=True)
    
    groq_api_key_encrypted = Column(Text, nullable=True)  
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    
    courses = relationship("Course", back_populates="user", cascade="all, delete-orphan")
    files = relationship("File", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.email}>"


class Course(Base):
    """Course model - synced from Google Classroom."""
    
    __tablename__ = "courses"
    
    id = Column(String(36), primary_key=True)  
    
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    classroom_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    section = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    room = Column(String(255), nullable=True)
    owner_id = Column(String(255), nullable=True)
    
    last_synced = Column(DateTime(timezone=True), nullable=True)
    sync_status = Column(String(50), default="pending")  
    sync_error = Column(Text, nullable=True)
    
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="courses")
    files = relationship("File", back_populates="course", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_courses_user_classroom", "user_id", "classroom_id", unique=True),
    )
    
    def __repr__(self):
        return f"<Course {self.name}>"


class File(Base):
    """File model - tracks downloaded and processed files."""
    
    __tablename__ = "files"
    
    id = Column(String(36), primary_key=True)  
    
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(36), ForeignKey("courses.id", ondelete="CASCADE"), nullable=True)
    
    drive_id = Column(String(255), nullable=True, index=True)
    drive_name = Column(String(512), nullable=False)
    mime_type = Column(String(255), nullable=True)
    web_view_link = Column(String(1024), nullable=True)
    
    local_path = Column(String(1024), nullable=True)
    file_size = Column(BigInteger, nullable=True)  # 64-bit: supports files > 2.1 GB
    file_hash = Column(String(64), nullable=True)  
    
    processing_status = Column(String(50), default="pending")  
    processing_error = Column(Text, nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    vector_store_id = Column(String(255), nullable=True)
    chunk_count = Column(Integer, default=0)
    contains_visual = Column(Boolean, default=False)
    
    detected_type = Column(String(50), nullable=True)  
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_deleted = Column(Boolean, default=False)  
    
    user = relationship("User", back_populates="files")
    course = relationship("Course", back_populates="files")
    
    __table_args__ = (
        Index("ix_files_user_drive", "user_id", "drive_id", unique=True),
        Index("ix_files_processing_status", "processing_status"),
    )
    
    def __repr__(self):
        return f"<File {self.drive_name}>"
