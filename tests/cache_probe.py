# SPDX-License-Identifier: Apache-2.0
"""Synthetic, persisted requests used before and after SSD-cache restarts."""
import argparse
import concurrent.futures
import json
import re
import secrets
import time
import urllib.request
from pathlib import Path

ROOT = None
BASE = None
MODEL = None


def post(path, body, timeout=1200):
    return urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                   {'Content-Type': 'application/json'}), timeout=timeout)


def metrics():
    raw = urllib.request.urlopen(BASE + '/metrics', timeout=15).read().decode()
    return {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1]) for line in raw.splitlines()
            if line.startswith('vllm:') and not line.startswith('#')
            and any(s in line for s in ('prefix_cache', 'kv_offload', 'spec_decode', 'num_requests'))
            and not '_created' in line}


def request(body, label):
    before = metrics()
    start = time.monotonic()
    first = last = None
    text = ''
    usage = finish = None
    with post('/v1/chat/completions', {**body, 'stream': True, 'stream_options': {'include_usage': True}}) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            assert not event.get('error'), event
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                part = choice.get('delta', {}).get('content') or ''
                if part:
                    last = time.monotonic()
                    first = first or last
                text += part
                finish = choice.get('finish_reason') or finish
    assert first and usage
    result = {'label': label, 'content': text, 'finish_reason': finish, 'usage': usage,
              'ttft_s': first - start, 'elapsed_s': time.monotonic() - start,
              'metric_delta': {k: v - before.get(k, 0) for k, v in metrics().items()}}
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (out / (label + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'metric_delta'}), flush=True)
    return result


def prepare():
    if (ROOT / 'fixtures.json').exists():
        raise FileExistsError('Keep fixtures for replay; choose another --work-dir for a new cold run')
    suffix = secrets.token_hex(4).upper()
    fixtures = {}
    for name, target in [('32k', 32768), ('128k', 131072)]:
        nonce = 'SSD-V41-' + name.upper() + '-' + suffix
        count = target // 30
        for _ in range(6):
            prompt = f'Remember this retrieval identifier exactly: {nonce}.\n'
            prompt += ''.join(f'Record {i:07d}: The field team inspected the sensor, recorded the temperature, checked the cable, and filed a routine maintenance report.\n' for i in range(count))
            prompt += '\nReturn only the retrieval identifier from the beginning.'
            messages = [{'role': 'user', 'content': prompt}]
            body = {'model': MODEL, 'messages': messages,
                    'chat_template_kwargs': {'enable_thinking': False}}
            with post('/tokenize', {**body, 'add_generation_prompt': True}) as response:
                tokens = json.load(response)['count']
            if target - 512 <= tokens <= target:
                fixtures[name] = {'nonce': nonce, 'tokens': tokens,
                                  'body': {**body, 'temperature': 0, 'seed': 812,
                                           'max_tokens': 48}}
                break
            count = max(1, count + int((target - 128 - tokens) / (tokens / count)))
        else:
            raise RuntimeError((name, tokens))
    (ROOT / 'fixtures.json').write_text(json.dumps(fixtures) + '\n')
    print(json.dumps({k: {'tokens': v['tokens'], 'nonce': v['nonce']} for k, v in fixtures.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'concurrent'])
    parser.add_argument('--label')
    parser.add_argument('--case', choices=['32k', '128k', 'all'], default='all')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000',
                        help='Server origin, without /v1; /metrics and /tokenize must be reachable')
    parser.add_argument('--model', required=True)
    parser.add_argument('--work-dir', type=Path, default=Path('probe-results'))
    args = parser.parse_args()
    ROOT = args.work_dir
    ROOT.mkdir(parents=True, exist_ok=True)
    BASE = args.base_url.rstrip('/')
    MODEL = args.model
    if args.mode == 'prepare':
        prepare()
    else:
        assert args.label
        fixtures = json.loads((ROOT / 'fixtures.json').read_text())
        cases = list(fixtures) if args.case == 'all' else [args.case]
        def run_case(name):
            fixture = fixtures[name]
            result = request(fixture['body'], args.label + '-' + name)
            assert result['finish_reason'] == 'stop' and result['content'].strip() == fixture['nonce'], result
        if args.mode == 'concurrent':
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
                list(pool.map(run_case, cases))
        else:
            for case in cases:
                run_case(case)
