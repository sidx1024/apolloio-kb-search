from werkzeug.middleware.profiler import ProfilerMiddleware
from api.index_old import app

app.config['PROFILE'] = True
app.wsgi_app = ProfilerMiddleware(
    app.wsgi_app, restrictions=[30], profile_dir='.')
app.run(debug=False, host='0.0.0.0', port=3002)
