"""Offline unit tests: python -m unittest discover -s tests"""
import math
import unittest
from datetime import date, datetime, timedelta

from gann_timer.gann_engine import AnalysisSettings, GannEngine, route_triggers, sq9_levels, wrap180
from gann_timer.horizons_client import BodySeries, Ephemeris, parse_result
from gann_timer.instruments import (INSTRUMENT_CHOICES, SEGMENTS, ExcludedInstrumentError, assert_allowed,
                                    is_excluded, segments_for_choice, symbols_for)
from gann_timer.timeconv import align_to_session, is_nifty_expiry, us_dst_active, ut_to_ist


def synthetic(start, hours, rates, offsets):
    times = [start + timedelta(hours=h) for h in range(hours)]
    bodies = {}
    for name, rate in rates.items():
        s = BodySeries(name, times=list(times))
        s.lon = [(offsets[name] + rate * h / 24) % 360 for h in range(hours)]
        s.lat = [0.0] * hours; s.ra = list(s.lon); s.dec = [0.0] * hours
        bodies[name] = s
    return Ephemeris(times[0], times[-1], 1, bodies)


class TimeTests(unittest.TestCase):
    def test_ut_to_ist_offset(self):
        self.assertEqual(ut_to_ist(datetime(2026, 1, 1, 0, 0)).strftime("%Y-%m-%d %H:%M"), "2026-01-01 05:30")
        self.assertEqual(ut_to_ist(datetime(2026, 1, 1, 20, 0)).strftime("%Y-%m-%d %H:%M"), "2026-01-02 01:30")

    def test_sessions(self):
        # Tue 2026-10-27 05:00 UT = 10:30 IST -> live NSE
        self.assertEqual(align_to_session("NSE", datetime(2026, 10, 27, 5, 0))[0], "LIVE")
        # 12:00 UT = 17:30 IST: NSE closed, MCX live
        st, nxt = align_to_session("NSE", datetime(2026, 10, 27, 12, 0))
        self.assertEqual(st, "POST-CLOSE"); self.assertEqual(nxt.strftime("%Y-%m-%d %H:%M"), "2026-10-28 09:15")
        self.assertEqual(align_to_session("MCX", datetime(2026, 10, 27, 12, 0))[0], "LIVE")
        # Saturday -> next Monday open
        st, nxt = align_to_session("NSE", datetime(2026, 10, 31, 6, 0))
        self.assertEqual(st, "NON-TRADING DAY"); self.assertEqual(nxt.date(), date(2026, 11, 2))

    def test_mcx_dst_close(self):
        self.assertTrue(us_dst_active(date(2026, 7, 1)))
        self.assertFalse(us_dst_active(date(2026, 12, 1)))
        # 23:45 IST in December (US standard time) is still live on MCX
        self.assertEqual(align_to_session("MCX", datetime(2026, 12, 1, 18, 15))[0], "LIVE")
        self.assertEqual(align_to_session("MCX", datetime(2026, 7, 1, 18, 15))[0], "POST-CLOSE")

    def test_nifty_expiry_tuesday(self):
        self.assertTrue(is_nifty_expiry(date(2026, 10, 27)))
        self.assertFalse(is_nifty_expiry(date(2026, 10, 29)))


class ExclusionTests(unittest.TestCase):
    def test_no_gold_silver_anywhere(self):
        for seg in SEGMENTS.values():
            self.assertFalse(any(is_excluded(s) for s in seg.symbols))
        for choice in INSTRUMENT_CHOICES:
            self.assertFalse(any(is_excluded(s) for s in symbols_for(segments_for_choice(choice))))

    def test_guard_raises(self):
        for bad in ("GOLD", "GOLDM", "SILVERMIC", "goldpetal"):
            with self.assertRaises(ExcludedInstrumentError):
                assert_allowed(["COPPER", bad])


class GannTests(unittest.TestCase):
    def test_sq9(self):
        lo, hi = sq9_levels(24350, 0.0)
        self.assertLessEqual(lo, 24350); self.assertGreaterEqual(hi, 24350)
        self.assertTrue(math.isclose(math.sqrt(hi) - math.sqrt(lo), 2.0))
        # Authentic Gann Square-of-Nine odd-square base axis (1, 9, 25, 49 -> odd roots)
        self.assertAlmostEqual(math.sqrt(lo) % 2, 1.0, places=9)

    def test_conjunction_timing(self):
        # Mars at 1°/day starting 0°, Saturn fixed at 10° -> conjunction exactly 10 days later
        eph = synthetic(datetime(2026, 1, 1), 24 * 20, {"Mars": 1.0, "Saturn": 0.0}, {"Mars": 0.0, "Saturn": 10.0})
        ev = [e for e in GannEngine(eph, AnalysisSettings(["Mars", "Saturn"])).run() if e.kind == "Aspect"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].time_ut, datetime(2026, 1, 11))
        self.assertIn("Conjunction", ev[0].title)

    def test_ingress_and_routing(self):
        eph = synthetic(datetime(2026, 3, 1), 24 * 10, {"Venus": 1.2, "Neptune": 0.01},
                        {"Venus": 357.0, "Neptune": 100.0})
        evs = GannEngine(eph, AnalysisSettings(["Venus", "Neptune"])).run()
        ing = [e for e in evs if e.kind == "Ingress"]
        self.assertEqual(len(ing), 1); self.assertIn("Aries", ing[0].title)
        rows = route_triggers(ing, segments_for_choice("MCX (Energy + Base Metals)"), 750.0, None)
        # Venus rules copper -> base metals only; energy rows filtered by rulership
        self.assertTrue(rows and all(r.segment.key == "MCX_METALS" for r in rows))
        self.assertIn("COPPER", rows[0].symbols)

    def test_wrap(self):
        self.assertEqual(wrap180(359), -1); self.assertEqual(wrap180(181), -179)


class MidpointHarmonicTests(unittest.TestCase):
    def test_midpoint_engine_methods(self):
        from gann_timer.midpoint_engine import (
            analyze_midpoints_v2,
            calculate_mumbai_solar_noon,
            equation_of_time,
            format_circle_fraction,
        )

        d = date(2026, 10, 28)

        # Method 1: NSE Session
        res1 = analyze_midpoints_v2(d, "NSE_SESSION")
        self.assertEqual(res1.midpoint_time.strftime("%H:%M:%S"), "12:22:30")
        self.assertAlmostEqual(res1.minutes_per_degree, 375.0 / 360.0, places=4)

        # Method 3: Solar Noon (NOAA EoT accuracy)
        eot = equation_of_time(301)  # late Oct
        self.assertTrue(15.0 < eot < 17.0)
        noon = calculate_mumbai_solar_noon(d)
        self.assertEqual(noon.hour, 12)
        self.assertTrue(20 <= noon.minute <= 25)

        # Method 6: Price Equilibrium & Dual Harmonics
        res6 = analyze_midpoints_v2(d, "PRICE_50", price_high=24450.0, price_low=24250.0)
        self.assertEqual(res6.price_equilibrium, 24350.0)
        self.assertEqual(res6.price_range, 200.0)

        # Harmonic point at 90 deg:
        # Range support: 24350 - 200/4 = 24300
        # Range resist: 24350 + 200/4 = 24400
        p90 = [p for p in res6.harmonic_points if p.degree == 90.0][0]
        self.assertEqual(p90.fraction_ratio, "1/4")
        self.assertAlmostEqual(p90.price_left, 24300.0, places=2)
        self.assertAlmostEqual(p90.price_right, 24400.0, places=2)
        self.assertAlmostEqual(p90.sq9_left, (math.sqrt(24350.0) - 0.5) ** 2, places=2)
        self.assertAlmostEqual(p90.sq9_right, (math.sqrt(24350.0) + 0.5) ** 2, places=2)

        # Fraction tests (Tharun project canonical standards)
        self.assertEqual(format_circle_fraction(0.0), "0 (Center)")
        self.assertEqual(format_circle_fraction(45.0), "1/8")
        self.assertEqual(format_circle_fraction(90.0), "1/4")
        self.assertEqual(format_circle_fraction(120.0), "1/3")
        self.assertEqual(format_circle_fraction(135.0), "3/8")
        self.assertEqual(format_circle_fraction(180.0), "1/2")


class ParserTests(unittest.TestCase):
    def test_parse(self):
        txt = ("hdr\n$$SOE\n 2026-Jan-01 00:00, , , 283.879610807, -23.720007274,  282.6881475, -0.8911035,\n"
               "$$EOE\n")
        s = parse_result(txt, "Mars")
        self.assertEqual(s.times[0], datetime(2026, 1, 1)); self.assertAlmostEqual(s.lon[0], 282.6881475)


if __name__ == "__main__":
    unittest.main()
