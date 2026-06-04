# Campus Recommender Backend

FastAPI backend wrapper for `src.service.CampusRecommendService`.

## Install Dependencies

```powershell
D:\Anaconda\envs\bigdata\python.exe -m pip install fastapi uvicorn pydantic
```

## Start Server

```powershell
$env:PYTHONIOENCODING='utf-8'; D:\Anaconda\envs\bigdata\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

## Test Commands

Health check:

```powershell
curl http://127.0.0.1:8000/api/health
```

Personalized recommendation:

```powershell
curl "http://127.0.0.1:8000/api/recommend/u_0001?top_n=10"
```

Hot posts:

```powershell
curl "http://127.0.0.1:8000/api/posts/hot?top_n=5"
```

Post list:

```powershell
curl "http://127.0.0.1:8000/api/posts?page=1&page_size=5"
```

Post detail:

```powershell
curl "http://127.0.0.1:8000/api/posts/p_ext_000001"
```

User profile:

```powershell
curl "http://127.0.0.1:8000/api/users/u_0001/profile"
```

Boards:

```powershell
curl "http://127.0.0.1:8000/api/boards"
```

Tags:

```powershell
curl "http://127.0.0.1:8000/api/tags"
```

Record behavior, PowerShell one-line:

```powershell
curl -X POST "http://127.0.0.1:8000/api/behavior" -H "Content-Type: application/json" -d "{\"user_id\":\"demo_user_A\",\"post_id\":\"2101899312\",\"action_type\":\"view\",\"dwell_time\":20,\"context\":{\"time_period\":\"晚上\"}}"
```

Refresh user profile:

```powershell
curl -X POST "http://127.0.0.1:8000/api/users/u_0001/refresh-profile"
```

All APIs return:

```json
{
  "success": true,
  "message": "ok",
  "data": {}
}
```

Errors are returned as:

```json
{
  "success": false,
  "message": "error reason",
  "data": null
}
```
