"""从 index.html 里的计划数据生成系统日历文件。

计划改了之后重新运行：python3 study-calendar/make_ics.py
生成的 .ics 双击导入苹果日历即可；同一事件的 UID 不变，重新导入会更新而不是重复。
"""
import datetime
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
OUT = HERE / '阶段0-学习日历.ics'
TRACKER = 'https://claude.ai/artifact/DNdrzbZ41NYggPAc6wzkau'
ALARM_MIN = 15

html = (HERE / 'index.html').read_text(encoding='utf-8')
P = json.loads(re.search(r'<script type="application/json" id="plan">(.*?)</script>', html, re.S).group(1))
S = P['slots']


def to_min(s):
    return int(s[:2]) * 60 + int(s[3:])


def slots(date, minutes):
    """和页面上的 slot() 同一套规则：工作日晚上一段；周六上午最多 satMorningMax 分钟，超出的放到下午。"""
    if date.weekday() != 5:
        start = to_min(S['weekday'])
        return [(start, start + minutes, '')]
    am, pm, cap = to_min(S['satMorning']), to_min(S['satAfternoon']), S['satMorningMax']
    if minutes <= cap:
        return [(am, am + minutes, '')]
    return [(am, am + cap, '（上午）'), (pm, pm + minutes - cap, '（下午）')]


def esc(s):
    return s.replace('\\', '\\\\').replace(';', '\\;').replace(',', '\\,').replace('\n', '\\n')


def fold(line):
    # RFC 5545：每行不超过 75 个字节，续行以空格开头；按字符切，不拆开多字节的中文
    out, cur, n = [], '', 0
    for ch in line:
        b = len(ch.encode('utf-8'))
        if n + b > 75:
            out.append(cur)
            cur, n = ' ', 1
        cur += ch
        n += b
    out.append(cur)
    return '\r\n'.join(out)


def stamp(date, minute):
    return f'{date:%Y%m%d}T{minute // 60:02d}{minute % 60:02d}00'


weeks = [(datetime.date.fromisoformat(w['mon']), i + 1, w) for i, w in enumerate(P['weeks'])]
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
lines = [
    'BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//ai-studeam//study-calendar//CN', 'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH', 'X-WR-CALNAME:AI 学习', 'X-WR-TIMEZONE:Asia/Shanghai',
    'BEGIN:VTIMEZONE', 'TZID:Asia/Shanghai',
    'BEGIN:STANDARD', 'DTSTART:19700101T000000', 'TZOFFSETFROM:+0800', 'TZOFFSETTO:+0800', 'TZNAME:CST', 'END:STANDARD',
    'END:VTIMEZONE',
]
events = total = 0
for key, day in sorted(P['days'].items()):
    date = datetime.date.fromisoformat(key)
    assert date.weekday() != 6, f'{key} 是主日，不排学习'
    mon, no, week = next(w for w in weeks if w[0] <= date < w[0] + datetime.timedelta(days=7))
    minutes = sum(t[1] for t in day['tasks'])
    total += minutes
    status = '草稿，周六复盘后在 ai-studeam 里重排' if week['draft'] else '已排进 ai-studeam'
    desc = '\n'.join([
        f'第 {no} 周 · {week["name"]}（{status}）',
        f'本周目标：{week["goal"]}',
        '',
        f'今天 {minutes / 60:g} 小时：',
        *[f'· {t}（{m} 分钟）' for t, m, _ in day['tasks']],
        '',
        f'勾任务、计时、做测验：{TRACKER}',
    ])
    for k, (start, end, part) in enumerate(slots(date, minutes)):
        lines += [
            'BEGIN:VEVENT',
            f'UID:{key}-{k}@ai-studeam.study-calendar',
            f'DTSTAMP:{now}',
            f'DTSTART;TZID=Asia/Shanghai:{stamp(date, start)}',
            f'DTEND;TZID=Asia/Shanghai:{stamp(date, end)}',
            f'SUMMARY:{esc("阶段0 · " + day["title"] + part)}',
            f'DESCRIPTION:{esc(desc)}',
            f'URL:{TRACKER}',
            'BEGIN:VALARM', 'ACTION:DISPLAY', f'DESCRIPTION:{esc(day["title"])}', f'TRIGGER:-PT{ALARM_MIN}M', 'END:VALARM',
            'END:VEVENT',
        ]
        events += 1
lines.append('END:VCALENDAR')

OUT.write_text('\r\n'.join(fold(l) for l in lines) + '\r\n', encoding='utf-8')
print(f'{OUT.name}：{len(P["days"])} 个学习日，{events} 条日历事件，共 {total / 60:g} 小时')
