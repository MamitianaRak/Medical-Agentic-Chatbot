import httpx
from app.config import settings

_agreed = set()


def run(model, payload, timeout=60):
    """Call a Workers AI model over REST. Returns the JSON `result` or raw bytes (e.g. audio/mpeg)."""
    if not settings.cloudflare_account_id or not settings.cloudflare_api_token:
        raise ValueError('CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set')
    url = f'https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}/ai/run/{model}'
    r = httpx.post(url, headers={'Authorization': f'Bearer {settings.cloudflare_api_token}'}, json=payload, timeout=timeout)
    if r.is_error:
        raise RuntimeError(f'Workers AI {model} failed ({r.status_code}): {r.text[:500]}')
    if r.headers.get('content-type', '').startswith('application/json'):
        return r.json()['result']
    return r.content


def accept_license(model):
    # Meta Llama 3.2 models require a one-time license acceptance per Cloudflare account.
    if model in _agreed:
        return
    try:
        run(model, {'prompt': 'agree'})
    except RuntimeError as exc:
        # Already-accepted accounts may reply with an error; the real call will surface real problems.
        print('License acceptance:', exc, flush=True)
    _agreed.add(model)
