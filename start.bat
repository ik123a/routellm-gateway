@echo off
echo ===================================================
echo   Starting RouteLLM-Gateway (Docker)
echo ===================================================
echo.
echo Bringing up the backend, mock server, and Redis cache...
echo.

docker-compose up --build -d

echo.
echo ===================================================
echo   SUCCESS! The system is now running.
echo ===================================================
echo.
echo Open your browser and navigate to:
echo    http://localhost:8000
echo.
echo To stop the server later, run: docker-compose down
echo.
pause
