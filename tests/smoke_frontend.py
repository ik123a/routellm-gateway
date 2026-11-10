"""Real-browser renderer security regression with a synthetic HTTP response."""
import functools
import http.server
import json
from pathlib import Path
import sys
import threading
from playwright.sync_api import sync_playwright

PAYLOAD = '<img src="/synthetic-missing-image" onerror="window.syntheticExecution=1">'


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith('/static/'):
            self.path = self.path.removeprefix('/static')
        super().do_GET()

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        data = {'choices': [{'message': {'content': PAYLOAD}}],
                '_routellm': {'predicted_tier': 'CHEAP', 'model_used': PAYLOAD}}
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    frontend = Path(__file__).resolve().parents[1] / 'frontend'
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), functools.partial(Handler, directory=str(frontend)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge' if sys.platform == 'win32' else None, headless=True)
            try:
                page = browser.new_page()
                page.route('**/*', lambda route: route.continue_()
                           if route.request.url.startswith(origin + '/')
                           or route.request.url.startswith('https://cdn.jsdelivr.net/npm/marked/')
                           else route.abort())
                page.goto(origin + '/', wait_until='networkidle')
                page.locator('label.switch').click()
                assert not page.locator('#stream-toggle').is_checked()
                page.locator('#prompt-input').fill(PAYLOAD)
                page.get_by_role('button', name='Send', exact=True).click()
                page.wait_for_function("!document.querySelector('#send-btn').disabled")
                print(json.dumps({'injected_image_nodes': page.locator('.message img').count(),
                                  'synthetic_script_executed': page.evaluate('window.syntheticExecution || 0')}), flush=True)
                assert page.locator('.user-message .message-content').inner_text() == PAYLOAD
                assert page.locator('.assistant-message .message-content').inner_text() == PAYLOAD
                assert page.locator('.routing-meta').inner_text().endswith(PAYLOAD)
                assert page.locator('.message img').count() == 0
                assert page.evaluate('window.syntheticExecution || 0') == 0
                for width in (320, 390, 768, 1440):
                    page.set_viewport_size({'width': width, 'height': 900})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'overflow at {width}px'
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print('User text, assistant text and model metadata render as inert text; no injected image or script execution.')


if __name__ == '__main__':
    main()
