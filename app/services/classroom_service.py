import asyncio
from typing import List, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import backoff

from app.core.exceptions import ClassroomAPIError, DriveAPIError


class ClassroomService:
    """Enhanced service for Google Classroom API with async support."""
    
    def __init__(self, credentials: Credentials):
        """
        Initialize service with Google credentials.
        
        Args:
            credentials: Valid Google Credentials object
        """
        self.credentials = credentials
        self.classroom_service = build('classroom', 'v1', credentials=credentials)
        self.drive_service = build('drive', 'v3', credentials=credentials)
    
    @backoff.on_exception(
        backoff.expo,
        HttpError,
        max_tries=3,
        giveup=lambda e: e.resp.status in [400, 401, 403, 404]
    )
    async def list_courses(self, page_size: int = 50) -> List[dict]:
        """
        List all courses for the user with pagination.
        
        Returns:
            List of course dictionaries
        """
        try:
            courses = []
            page_token = None
            
            while True:
                result = await asyncio.to_thread(
                    lambda pt=page_token: self.classroom_service.courses().list(
                        pageSize=page_size,
                        pageToken=pt
                    ).execute()
                )
                
                courses.extend(result.get('courses', []))
                page_token = result.get('nextPageToken')
                
                if not page_token:
                    break
            
            return courses
        
        except HttpError as e:
            raise ClassroomAPIError(f"Failed to list courses: {str(e)}")
    
    @backoff.on_exception(backoff.expo, HttpError, max_tries=3)
    async def get_course(self, course_id: str) -> dict:
        """Get detailed course information."""
        try:
            return await asyncio.to_thread(
                lambda: self.classroom_service.courses().get(id=course_id).execute()
            )
        except HttpError as e:
            raise ClassroomAPIError(f"Failed to get course {course_id}: {str(e)}")
    
    @backoff.on_exception(backoff.expo, HttpError, max_tries=3)
    async def list_coursework(self, course_id: str, page_size: int = 50) -> List[dict]:
        """List all coursework (assignments) for a course."""
        try:
            coursework = []
            page_token = None
            
            while True:
                result = await asyncio.to_thread(
                    lambda pt=page_token: self.classroom_service.courses().courseWork().list(
                        courseId=course_id,
                        pageSize=page_size,
                        pageToken=pt
                    ).execute()
                )
                
                coursework.extend(result.get('courseWork', []))
                page_token = result.get('nextPageToken')
                
                if not page_token:
                    break
            
            return coursework
        
        except HttpError as e:
            raise ClassroomAPIError(f"Failed to list coursework for {course_id}: {str(e)}")
    
    @backoff.on_exception(backoff.expo, HttpError, max_tries=3)
    async def list_coursework_materials(self, course_id: str, page_size: int = 50) -> List[dict]:
        """List all coursework materials for a course."""
        try:
            materials = []
            page_token = None
            
            while True:
                result = await asyncio.to_thread(
                    lambda pt=page_token: self.classroom_service.courses().courseWorkMaterials().list(
                        courseId=course_id,
                        pageSize=page_size,
                        pageToken=pt
                    ).execute()
                )
                
                materials.extend(result.get('courseWorkMaterial', []))
                page_token = result.get('nextPageToken')
                
                if not page_token:
                    break
            
            return materials
        
        except HttpError as e:
            raise ClassroomAPIError(f"Failed to list materials for {course_id}: {str(e)}")
    
    @backoff.on_exception(backoff.expo, HttpError, max_tries=3)
    async def list_announcements(self, course_id: str, page_size: int = 50) -> List[dict]:
        """List all announcements for a course (often contain lecture PDFs)."""
        try:
            announcements = []
            page_token = None
            
            while True:
                result = await asyncio.to_thread(
                    lambda pt=page_token: self.classroom_service.courses().announcements().list(
                        courseId=course_id,
                        pageSize=page_size,
                        pageToken=pt
                    ).execute()
                )
                
                announcements.extend(result.get('announcements', []))
                page_token = result.get('nextPageToken')
                
                if not page_token:
                    break
            
            return announcements
        
        except HttpError as e:
            raise ClassroomAPIError(f"Failed to list announcements for {course_id}: {str(e)}")
    
    async def extract_drive_files(self, coursework_or_material: dict) -> List[dict]:
        """
        Extract Drive file IDs and metadata from coursework/material/announcement.
        
        Returns:
            List of {
                "drive_id": ...,
                "drive_name": ...,
                "mime_type": ...,
                "web_view_link": ...
            }
        """
        files = []
        
        materials = coursework_or_material.get('materials', [])
        
        for material in materials:
            if 'driveFile' in material:
                drive_file = material['driveFile']['driveFile']
                files.append({
                    "drive_id": drive_file['id'],
                    "drive_name": drive_file.get('title', 'Untitled'),
                    "mime_type": drive_file.get('mimeType'),
                    "web_view_link": drive_file.get('alternateLink')
                })
            
            elif 'link' in material:
                link = material['link']
                files.append({
                    "drive_id": f"link_{hash(link['url'])}",
                    "drive_name": link.get('title', 'Link'),
                    "mime_type": "text/html",
                    "web_view_link": link.get('url')
                })
        
        return files
    
    async def get_all_course_files(self, course_id: str) -> List[dict]:
        """
        Get all files from coursework, materials, AND announcements.
        
        Google Classroom stores files in 3 places:
          1. courseWork (assignments) — e.g., "Assignment 1", "LAB 1"
          2. courseWorkMaterials — e.g., shared resources
          3. announcements — e.g., "Lecture 1-5 Python.pdf" posted via class posts
        
        Returns:
            List of unique Drive files from all sources.
        """
        all_files = []
        seen_ids = set()
        
        # Source 1: Assignments (courseWork)
        coursework = await self.list_coursework(course_id)
        for work in coursework:
            files = await self.extract_drive_files(work)
            for file in files:
                if file['drive_id'] not in seen_ids:
                    all_files.append(file)
                    seen_ids.add(file['drive_id'])
        
        # Source 2: Course materials
        materials = await self.list_coursework_materials(course_id)
        for material in materials:
            files = await self.extract_drive_files(material)
            for file in files:
                if file['drive_id'] not in seen_ids:
                    all_files.append(file)
                    seen_ids.add(file['drive_id'])
        
        # Source 3: Announcements (often contain lecture PDFs!)
        # Gracefully degrade if announcements scope not yet granted
        try:
            announcements = await self.list_announcements(course_id)
            for announcement in announcements:
                files = await self.extract_drive_files(announcement)
                for file in files:
                    if file['drive_id'] not in seen_ids:
                        all_files.append(file)
                        seen_ids.add(file['drive_id'])
        except (ClassroomAPIError, HttpError) as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Could not fetch announcements for {course_id} "
                f"(user may need to re-login to grant announcements scope): {e}"
            )
        
        return all_files