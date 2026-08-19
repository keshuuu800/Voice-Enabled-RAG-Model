"""
Convenience script to start the HH-Goa Voice RAG server.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 8000 --reload
"""
import argparse
import uvicorn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HH-Goa Voice RAG Server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--log-level', default='info')
    args = parser.parse_args()
    
    print(f'Starting HH-Goa Voice RAG at http://{args.host}:{args.port}')
    print(f'Frontend: http://localhost:{args.port}')
    print(f'API Docs: http://localhost:{args.port}/docs')
    print()
    
    uvicorn.run(
        'app.main:app',
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
