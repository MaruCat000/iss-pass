#!/usr/bin/env python3
"""
国際宇宙ステーション(ISS)が自分の上空を通る時刻を予測し、肉眼で見える回を当てる。

2026-11-03 作成
  きっかけ: 気分転換の雑談で、starlink_orbits.py の続きとして「ISS が自分の上を通る時刻を当てたい」
  方針    : starlink_orbits.py と同じく軌道計算は sgp4 に任せ、座標変換は自前の数行で済ませる。
            「見える」の条件は次の3つを全部満たすこと。
              1. ISS が地平線から10度以上の高さにいる
              2. 観測地点の空が暗い(太陽が地平線の6度以上下 = 市民薄明の終わり以降)
              3. ISS 自身には太陽が当たっている(地球の影に入っていない)
            ISS は自分で光らず、太陽の反射で光るので、3 がないと「上にいるのに見えない」回を拾ってしまう。
  やらないこと: 明るさ(等級)の予測はしない。精度は数十秒〜1分程度で、見上げる目安としては十分。
            正確な予報は JAXA「#きぼうを見よう」や Heavens-Above で答え合わせする。
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, jday

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle"  # 25544 = ISS
COAST_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"
UA = {"User-Agent": "iss-pass-sample/1.0"}
JST = timezone(timedelta(hours=9))
WEEK = "月火水木金土日"
EARTH_R = 6371.0
WGS84_A, WGS84_F = 6378.137, 1 / 298.257223563
MIN_ELEV = 10.0          # これより低いと建物や山に隠れて実際は見えにくい
DARK_SUN_ELEV = -6.0     # 太陽がこれより下なら空が十分暗い
DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
        "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]


# ---------- データ取得 ----------

def fetch(url, path, max_age_h):
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_h * 3600
    if not fresh:
        print("fetching:", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_iss(refresh):
    # ISS はときどき軌道を噴射で持ち上げるので、古い TLE だと予測がずれる。6時間で取り直す
    lines = [ln.rstrip() for ln in fetch(TLE_URL, "iss.tle", 0 if refresh else 6).splitlines() if ln.strip()]
    if len(lines) < 3 or not lines[1].startswith("1 "):
        raise SystemExit("ISS の TLE を読めませんでした")
    return Satrec.twoline2rv(lines[1], lines[2]), lines[1]


# ---------- 座標計算 ----------

def gmst_rad(jd):
    """グリニッジ平均恒星時(ラジアン)。jd は配列可。"""
    t = (jd - 2451545.0) / 36525.0
    sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * t + 0.093104 * t ** 2 - 6.2e-6 * t ** 3
    return np.radians((sec % 86400.0) / 240.0)


def rot_z(v, theta):
    """[N,3] の各ベクトルを z 軸まわりに -theta 回す(慣性系 → 地球固定系)。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)


def sun_eci(jd):
    """太陽の方向(単位ベクトル, 慣性系)。天文年鑑の簡易式で、精度は0.01度程度。"""
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=1)


def observer_ecef(lat, lon, h_km):
    la, lo = math.radians(lat), math.radians(lon)
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return np.array([(n + h_km) * math.cos(la) * math.cos(lo),
                     (n + h_km) * math.cos(la) * math.sin(lo),
                     (n * (1 - e2) + h_km) * math.sin(la)])


def enu_matrix(lat, lon):
    """地球固定系 → 観測地点の「東・北・上」への回転行列。"""
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([[-math.sin(lo), math.cos(lo), 0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def elev_azim(enu):
    e, n, u = enu[:, 0], enu[:, 1], enu[:, 2]
    elev = np.degrees(np.arctan2(u, np.hypot(e, n)))
    azim = np.degrees(np.arctan2(e, n)) % 360  # 北=0, 東=90
    return elev, azim


def compass(az):
    return DIRS[int((az + 11.25) // 22.5) % 16]


def compute(sat, start, hours, step_s, lat, lon, h_km):
    """start から hours 時間ぶん、step_s 秒刻みで ISS の見え方を計算する。"""
    ts = np.arange(0, hours * 3600 + 1, step_s, dtype=float)
    t0 = start.astimezone(timezone.utc)
    jd0, fr0 = jday(t0.year, t0.month, t0.day, t0.hour, t0.minute, t0.second + t0.microsecond / 1e6)
    jd, fr = np.full_like(ts, jd0), fr0 + ts / 86400.0
    err, r, _v = sat.sgp4_array(jd, fr)
    if np.any(err):
        print(f"warning: sgp4 error at {int(np.count_nonzero(err))} steps")
    jdf = jd + fr
    theta = gmst_rad(jdf)
    r_ecef = rot_z(r, theta)
    sun_dir_eci = sun_eci(jdf)
    sun_ecef = rot_z(sun_dir_eci, theta)

    obs = observer_ecef(lat, lon, h_km)
    M = enu_matrix(lat, lon)
    iss_el, iss_az = elev_azim((r_ecef - obs) @ M.T)
    sun_el, _ = elev_azim(sun_ecef @ M.T)            # 太陽は十分遠いので方向だけで高度が出る

    # 円柱近似の地球の影: 太陽と反対側にいて、かつ太陽方向の軸から地球半径以内なら影の中
    along = np.sum(r * sun_dir_eci, axis=1)
    perp = np.linalg.norm(r - along[:, None] * sun_dir_eci, axis=1)
    sunlit = (along > 0) | (perp > EARTH_R)

    times = [t0 + timedelta(seconds=s) for s in ts]
    lat_g = np.degrees(np.arctan2(r_ecef[:, 2], np.hypot(r_ecef[:, 0], r_ecef[:, 1])))
    lon_g = np.degrees(np.arctan2(r_ecef[:, 1], r_ecef[:, 0]))
    return dict(times=times, el=iss_el, az=iss_az, sun_el=sun_el, sunlit=sunlit, lat=lat_g, lon=lon_g)


def find_passes(d):
    """地平線から MIN_ELEV 度以上にいる連続区間を1回の通過としてまとめる。"""
    up = d["el"] >= MIN_ELEV
    passes, i, n = [], 0, len(up)
    while i < n:
        if not up[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and up[j + 1]:
            j += 1
        idx = np.arange(i, j + 1)
        vis = idx[(d["sun_el"][idx] < DARK_SUN_ELEV) & d["sunlit"][idx]]
        # 見える回は「見えている間」の最高点を使う。通過全体の最高点は影の中のことがあるため
        span = vis if len(vis) else idx
        k = span[np.argmax(d["el"][span])]
        # 空の高いところで見えなくなる = 地球の影に入って消えた
        fades = len(vis) > 0 and d["el"][vis[-1]] >= MIN_ELEV + 5 and not d["sunlit"][min(vis[-1] + 1, n - 1)]
        passes.append(dict(idx=idx, vis=vis, top=k, visible=len(vis) > 0, fades=fades))
        i = j + 1
    return passes


# ---------- 描画 ----------

def plot_sky(d, p, out, place):
    """見える回の通り道を、真上を中心にした空の図で描く。外周が地平線、中心が天頂。"""
    fig = plt.figure(figsize=(7, 7.6), facecolor="#05070d")
    ax = fig.add_subplot(111, projection="polar", facecolor="#0b1626")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)  # 上が北、時計回りに東
    ax.set_rlim(0, 90); ax.set_rticks([30, 60, 80]); ax.set_yticklabels(["60°", "30°", "10°"], color="#8aa")
    ax.set_xticks(np.radians(range(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], color="white")
    ax.grid(color="#345", lw=0.6)

    idx = p["idx"]
    r = 90 - d["el"][idx]; th = np.radians(d["az"][idx])
    ax.plot(th, r, color="#556", lw=1.5, ls="--")                 # 見えない部分(影の中・空が明るい)
    v = p["vis"]
    if len(v):
        ax.plot(np.radians(d["az"][v]), 90 - d["el"][v], color="#ffd060", lw=3)
        for k, label in ((v[0], "start"), (v[-1], "end")):
            ax.plot(np.radians(d["az"][k]), 90 - d["el"][k], "o", color="#ffd060", ms=7)
            ax.annotate(f"{label} {d['times'][k].astimezone(JST):%H:%M:%S}",
                        (np.radians(d["az"][k]), 90 - d["el"][k]), color="white", fontsize=9,
                        xytext=(6, 6), textcoords="offset points")
    k = p["top"]
    ax.plot(np.radians(d["az"][k]), 90 - d["el"][k], "*", color="white", ms=12)
    t = d["times"][v[0] if len(v) else idx[0]].astimezone(JST)
    ax.set_title(f"ISS pass  {t:%Y-%m-%d %H:%M} JST  ({place})\n"
                 f"max elevation {d['el'][k]:.0f} deg   yellow = visible, dashed = not visible",
                 color="white", fontsize=11, pad=18)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("saved:", out)


def plot_ground(d, lat, lon, out, minutes):
    """ここから minutes 分の地上軌跡(ISS の真下の点の移動)を世界地図に描く。"""
    gj = json.loads(fetch(COAST_URL, "cache_coastline.geojson", 24 * 30))
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="black")
    ax.set_facecolor("#10263d")
    for f in gj["features"]:
        g = f["geometry"]
        for part in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            a = np.array(part)
            ax.plot(a[:, 0], a[:, 1], color="#7fa8c9", lw=0.6)
    n = sum(1 for t in d["times"] if (t - d["times"][0]).total_seconds() <= minutes * 60)
    lo, la, lit = d["lon"][:n].copy(), d["lat"][:n].copy(), d["sunlit"][:n]
    jump = np.where(np.abs(np.diff(lo)) > 180)[0] + 1        # 経度±180 をまたぐ所で線を切る
    lo_s, la_s = np.insert(lo, jump, np.nan), np.insert(la, jump, np.nan)
    ax.plot(lo_s, la_s, color="#556", lw=1.2)
    ax.scatter(lo[lit], la[lit], s=2, color="#ffd060", label="in sunlight")
    ax.plot(lo[0], la[0], "o", color="white", ms=9)
    ax.annotate(f"ISS now {d['times'][0].astimezone(JST):%H:%M} JST", (lo[0], la[0]), color="white",
                xytext=(8, 8), textcoords="offset points")
    ax.plot(lon, lat, "*", color="#ff6060", ms=14)
    ax.annotate("you", (lon, lat), color="#ff9090", xytext=(8, -14), textcoords="offset points")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(colors="white"); ax.grid(color="#334", lw=0.5)
    ax.legend(loc="lower left", facecolor="#10263d", labelcolor="white")
    ax.set_title(f"ISS ground track, next {minutes} min   data: CelesTrak, propagation: SGP4",
                 color="white")
    fig.savefig(out, dpi=130, facecolor="black", bbox_inches="tight")
    print("saved:", out)


# ---------- エントリポイント ----------

def main():
    ap = argparse.ArgumentParser(description="ISS が上空を通る時刻と、肉眼で見える回を予測する")
    ap.add_argument("--lat", type=float, default=35.6812, help="観測地点の緯度(既定: 東京駅)")
    ap.add_argument("--lon", type=float, default=139.7671, help="観測地点の経度(既定: 東京駅)")
    ap.add_argument("--height", type=float, default=0.0, help="観測地点の標高 m")
    ap.add_argument("--place", default="Tokyo", help="図に入れる地名(英字)")
    ap.add_argument("--days", type=float, default=5, help="何日先まで探すか")
    ap.add_argument("--all", action="store_true", help="見えない回(昼間・影の中)も一覧に出す")
    ap.add_argument("--track-min", type=int, default=93, help="地上軌跡を描く長さ(分)。ISS の1周は約93分")
    ap.add_argument("--refresh", action="store_true", help="TLE のキャッシュを無視して取り直す")
    args = ap.parse_args()

    sat, l1 = load_iss(args.refresh)
    epoch_age_h = (sat.jdsatepoch + sat.jdsatepochF - (time.time() / 86400 + 2440587.5)) * -24
    print(f"TLE の基準時刻: {epoch_age_h:.1f} 時間前")

    now = datetime.now(timezone.utc)
    d = compute(sat, now, args.days * 24, 10, args.lat, args.lon, args.height / 1000)
    passes = find_passes(d)
    shown = passes if args.all else [p for p in passes if p["visible"]]

    print(f"\n観測地点: 緯度 {args.lat}, 経度 {args.lon}   今から {args.days:g} 日間")
    print(f"上空を通る回: {len(passes)} 回 / うち肉眼で見える回: {sum(p['visible'] for p in passes)} 回\n")
    if not shown:
        print("この期間に見える回はありません。ISS は時期によって、夕方・明け方に見えない週が続きます。--days を延ばしてみてください")
    for p in shown:
        idx = p["vis"] if p["visible"] else p["idx"]
        a, b, k = idx[0], idx[-1], p["top"]
        ta, tb = d["times"][a].astimezone(JST), d["times"][b].astimezone(JST)
        mark = "★見える" if p["visible"] else "  見えない"
        end = "で影に入って消える" if p["fades"] else "へ沈む"
        print(f"{mark}  {ta:%m/%d}({WEEK[ta.weekday()]}) {ta:%H:%M:%S} {compass(d['az'][a])}の空から  →  "
              f"最高 {d['el'][k]:2.0f}度 {compass(d['az'][k])}  →  {tb:%H:%M:%S} {compass(d['az'][b])}{end}"
              f"  ({(tb - ta).seconds // 60}分{(tb - ta).seconds % 60:02d}秒)")

    vis = [p for p in passes if p["visible"]]
    if vis:
        plot_sky(d, vis[0], "iss_next_pass.png", args.place)
    plot_ground(d, args.lat, args.lon, "iss_ground_track.png", args.track_min)


if __name__ == "__main__":
    sys.exit(main())
