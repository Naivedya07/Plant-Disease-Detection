@echo off
echo Starting Plant Disease Detection Server...
cd /d "D:\6th Sem\Project - II\webapp"
uvicorn main:app --reload
pause