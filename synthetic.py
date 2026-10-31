"""Synthetic demo data generator (clearly fake) so the pipeline runs offline.
Injects: (1) a border-incident event on day 3, (2) a coordinated false-claim campaign,
(3) a sentiment shift after the event."""
import random
import pandas as pd
from datetime import datetime, timedelta

START = datetime(2026, 10, 25, 0, 0)

EN_NEUTRAL = ["Army conducts routine exercise near the {r} sector today", "Defence ministry briefing scheduled this afternoon",
              "Joint military drill begins, officials say it is routine", "Weather delays flights near the border region",
              "Parliament committee discusses defence budget allocation"]
EN_POS = ["Proud of our soldiers, salute to the brave jawans", "Great success for the forces, strong response and peace restored",
          "Our army is a hero, safe borders thanks to them", "Positive talks, peace process moves forward"]
EN_NEG = ["Terror attack condemned, fear in the region", "Ceasefire violation again, civilians scared and killed",
          "Shameful failure of intelligence, war like situation", "Attack on convoy leaves several killed, people angry"]
HI = ["हमारे सैनिकों पर गर्व है, सलाम", "सीमा पर हमला, लोग डरे हुए हैं, निंदा", "सेना का अभ्यास आज सामान्य रूप से जारी", "शांति और सफल वार्ता की उम्मीद"]
UR = ["سرحد پر حملہ، لوگ خوفزدہ ہیں", "فوجی مشق معمول کے مطابق جاری", "امن مذاکرات کامیاب ہونے کی امید"]
FALSE_CLAIM = ["BREAKING!!! Army base destroyed, 500 soldiers killed, media hiding the truth, share before deleted",
               "BREAKING: army base destroyed 500 soldiers killed!! media is hiding truth. share before they delete",
               "Breaking!!! Army base destroyed and 500 soldiers killed, media hiding truth, share before deleted!!"]
REGIONS = ["northern", "western", "eastern", "coastal"]


def generate(seed=7, n_organic=650):
    rng = random.Random(seed)
    rows = []
    pid = 0
    for _ in range(n_organic):
        t = START + timedelta(minutes=rng.randint(0, 5 * 24 * 60))
        day = (t - START).days
        country = rng.choices(["India", "Pakistan", "China"], [0.5, 0.3, 0.2])[0]
        lang = rng.choices(["en", "hi", "ur"], [0.6, 0.25, 0.15] if country == "India" else [0.55, 0.05, 0.4])[0]
        # sentiment shifts negative from day 3 (border incident)
        neg_p = 0.2 if day < 3 else 0.55
        r = rng.random()
        if lang == "en":
            pool = EN_NEG if r < neg_p else (EN_POS if r < neg_p + 0.25 else EN_NEUTRAL)
        elif lang == "hi":
            pool = [HI[1]] if r < neg_p else [HI[0], HI[3], HI[2]]
        else:
            pool = [UR[0]] if r < neg_p else [UR[1], UR[2]]
        text = rng.choice(pool).format(r=rng.choice(REGIONS))
        if rng.random() < 0.3:
            text += " " + rng.choice(["#IndiaArmy", "#Border", "#Security", "#Defence"])
        pid += 1
        rows.append(dict(post_id=pid, author=f"user_{rng.randint(1, 320)}", timestamp=t, country=country,
                         platform=rng.choice(["X", "Reddit", "Facebook"]), text=text,
                         likes=int(rng.expovariate(1 / 15)), shares=int(rng.expovariate(1 / 4)),
                         account_age_days=rng.randint(200, 3000), has_source_link=rng.random() < 0.25))
    # coordinated campaign: 18 new accounts, day 3-4, near-identical text, within minutes
    base = START + timedelta(days=3, hours=9)
    for i in range(18):
        pid += 1
        t = base + timedelta(minutes=rng.randint(0, 40))
        rows.append(dict(post_id=pid, author=f"patriot_news_{i}", timestamp=t, country="Pakistan",
                         platform="X", text=rng.choice(FALSE_CLAIM) + " " + rng.choice(["#Breaking", "#WarNews"]),
                         likes=rng.randint(80, 400), shares=rng.randint(150, 900),
                         account_age_days=rng.randint(2, 20), has_source_link=False))
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df
