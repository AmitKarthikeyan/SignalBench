# SignalBench

ML-powered stock signal backtesting platform.

## Project Structure

```
SignalBench/
├── backend/                # Backend service
│   ├── app/               # FastAPI application
│   │   ├── main.py       # Main application entry point
│   │   ├── ml/           # ML models and data fetching
│   │   │   └── data.py
│   │   ├── models.py     # Database models
│   │   └── config.py     # Configuration
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile        # Backend container
├── frontend/              # Frontend application
├── scripts/               # Utility scripts
├── docker-compose.yml     # Full stack orchestration
└── README.md
```

## Setup

### Local Development

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Run the backend:
```bash
cd backend
YF_USE_CURL_CFFI=1 uvicorn app.main:app --reload --port 8000
```

### Docker

1. Build and run all services:
```bash
docker-compose up --build
```

2. Access the application:
   - Backend API: http://localhost:8000
   - Frontend: http://localhost:5173
   - PostgreSQL: localhost:5432
   - Redis: localhost:6379

## Database

Connect to PostgreSQL:
```bash
docker-compose exec postgres psql -U app -d signalbench
```

View tables:
```sql
\dt
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

- `YF_USE_CURL_CFFI=1` - Required for yfinance to work properly
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string

