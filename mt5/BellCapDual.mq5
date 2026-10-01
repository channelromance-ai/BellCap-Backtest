//+------------------------------------------------------------------+
//| BellCapDual.mq5                                                  |
//|                                                                  |
//| Two legs in one expert, sharing one The5ers account:             |
//|                                                                  |
//|  EUR  short EURUSD at 22:00 Winnipeg (06:00 server), Sun-Thu     |
//|       nights, stop 40 pips above the fill, close 07:00 Winnipeg  |
//|       (15:00 server). 0.09 lots, scaled down by the share of the |
//|       6% loss allowance still left.          (eur_morning_late)  |
//|                                                                  |
//|  SPX  long SP500 for the server day: open at the first minutes   |
//|       after the 17:00 New York roll (01:00 server), close 15     |
//|       minutes before the roll or the session end, whichever is   |
//|       earlier. Size targets a 0.25% daily swing from the last 20 |
//|       daily moves (cap 3x), doubled on the day after a 2-sigma   |
//|       down day and on turn-of-month days, scaled by the share of |
//|       the 6% allowance left. Hard stop at 1% of the account.     |
//|                                       (index_prop_intraday.py)   |
//|                                                                  |
//| Server clock = New York + 7 = Winnipeg + 8, all year.            |
//| Before either leg opens, the worst case of every open stop plus  |
//| the new one must stay inside the daily loss limit.               |
//| Only positions with this expert's two magic numbers are touched. |
//+------------------------------------------------------------------+
#property copyright "BellCap-Backtest"
#property version   "2.00"

#include <Trade\Trade.mqh>

//--- account (The5ers one-step)
input bool   DryRun            = true;     // true = log only, place no trades
input double InitialBalance    = 5000.0;
input double TargetBalance     = 5500.0;   // stop trading once reached
input double FloorBalance      = 4700.0;   // overall loss level (static)
input double DailyLossLimit    = 150.0;    // daily loss, in $
input int    ServerMinusLocal  = 8;        // server hours ahead of Winnipeg

//--- EUR leg
input bool   EurEnabled        = true;
input string EurSymbol         = "EURUSD";
input double EurBaseLots       = 0.09;
input double EurStopPips       = 40.0;
input int    EurEntryHour      = 6;        // 06:00 server = 22:00 Winnipeg
input int    EurExitHour       = 15;       // 15:00 server = 07:00 Winnipeg
input double EurMaxSpreadPips  = 2.0;
input long   EurMagic          = 26092501;

//--- SPX leg
input bool   SpxEnabled        = true;
input string SpxSymbol         = "SP500";
input double SpxTargetVol      = 0.0025;   // target daily swing of account
input double SpxMaxExposure    = 3.0;      // cap on notional / equity
input double SpxBoost          = 2.0;      // dip-day / turn-of-month factor
input double SpxDipSigma       = 2.0;
input double SpxStopShare      = 0.01;     // stop = 1% of InitialBalance
input int    SpxEntryHour      = 1;        // 01:00 server = 17:00 Winnipeg
input int    SpxCloseHour      = 23;       // close at 23:45 server latest
input int    SpxCloseMinute    = 45;
input double SpxMaxSpreadPts   = 2.0;      // index points
input long   SpxMagic          = 26092502;

input int    MaxLateMinutes    = 30;       // never enter later than this
input double SlippageAllowPips = 2.0;

CTrade trade;
string LOG_FILE = "BellCapDual_log.csv";
string HB_FILE  = "BellCapDual_heartbeat.txt";
string GV_EUR   = "BCD_eur_day";
string GV_SPX   = "BCD_spx_day";

//+------------------------------------------------------------------+
datetime DayStart(datetime t)
  {
   MqlDateTime d;
   TimeToStruct(t, d);
   d.hour = 0; d.min = 0; d.sec = 0;
   return StructToTime(d);
  }

double PipSize(string sym)
  {
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double pt = SymbolInfoDouble(sym, SYMBOL_POINT);
   return (digits == 3 || digits == 5) ? 10 * pt : pt;
  }

//--- dollars per 1.0 price move per lot
double ValuePerPrice(string sym)
  {
   double tv = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   return ts > 0 ? tv / ts : 0.0;
  }

double RoundLots(string sym, double lots)
  {
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double mn = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double mx = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   if(step <= 0)
      step = 0.01;
   double v = MathFloor(lots / step + 1e-9) * step;
   if(v < mn)
      return 0.0;
   return MathMin(v, mx);
  }

//--- share of the 6% allowance still left, 0..1
double CushionShare()
  {
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double allowance = InitialBalance - FloorBalance;
   if(eq <= 0 || allowance <= 0)
      return 0.0;
   return MathMax(0.0, MathMin(1.0, (eq - FloorBalance) /
                                     (eq * allowance / InitialBalance)));
  }

double ClosedToday(datetime now)
  {
   double sum = 0;
   if(!HistorySelect(DayStart(now), now + 60))
      return 0;
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
     {
      ulong tk = HistoryDealGetTicket(i);
      long type = HistoryDealGetInteger(tk, DEAL_TYPE);
      if(type != DEAL_TYPE_BUY && type != DEAL_TYPE_SELL)
         continue;
      sum += HistoryDealGetDouble(tk, DEAL_PROFIT)
             + HistoryDealGetDouble(tk, DEAL_COMMISSION)
             + HistoryDealGetDouble(tk, DEAL_SWAP);
     }
   return sum;
  }

//--- further loss from here if every open position's stop were hit (any
//--- magic). Current open losses are already inside equity, so only the
//--- distance from the current price to each stop is added. A position
//--- without a stop adds nothing measurable and is reported once.
double OpenStopRisk()
  {
   double risk = 0;
   static datetime warned = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i) == 0)
         continue;
      string sym = PositionGetString(POSITION_SYMBOL);
      double sl = PositionGetDouble(POSITION_SL);
      double cur = PositionGetDouble(POSITION_PRICE_CURRENT);
      double vol = PositionGetDouble(POSITION_VOLUME);
      if(sl <= 0)
        {
         if(warned != DayStart(TimeTradeServer()))
           {
            Log("ALL", "WARNING", StringFormat("open %s position has no "
                                               "stop: its risk cannot be "
                                               "counted", sym));
            warned = DayStart(TimeTradeServer());
           }
         continue;
        }
      risk += MathAbs(cur - sl) * ValuePerPrice(sym) * vol;
     }
   return risk;
  }

//--- would a new worst-case loss of `extra` keep today inside the limit?
bool DailyRoomFor(double extra, datetime now, double &projected)
  {
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double start_bal = bal - ClosedToday(now);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   projected = start_bal - (eq - OpenStopRisk() - extra);
   return projected < DailyLossLimit * 0.95;
  }

//+------------------------------------------------------------------+
void Log(string leg, string event, string detail, double lots = 0,
         double price = 0, double sl = 0, double pnl = 0, double comm = 0)
  {
   datetime srv = TimeTradeServer();
   datetime loc = srv - ServerMinusLocal * 3600;
   int h = FileOpen(LOG_FILE, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI |
                    FILE_COMMON | FILE_SHARE_READ, ',');
   if(h == INVALID_HANDLE)
     {
      Print("BCD log open failed: ", GetLastError());
      return;
     }
   if(FileSize(h) == 0)
      FileWrite(h, "server_time", "winnipeg_time", "leg", "event", "detail",
                "lots", "price", "stop", "pnl", "commission", "balance",
                "equity", "dry_run");
   FileSeek(h, 0, SEEK_END);
   FileWrite(h, TimeToString(srv, TIME_DATE | TIME_SECONDS),
             TimeToString(loc, TIME_DATE | TIME_SECONDS), leg, event, detail,
             DoubleToString(lots, 2), DoubleToString(price, 5),
             DoubleToString(sl, 5), DoubleToString(pnl, 2),
             DoubleToString(comm, 2),
             DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
             DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2),
             DryRun ? "yes" : "no");
   FileClose(h);
   Print("BCD ", leg, " ", event, ": ", detail);
  }

void Heartbeat()
  {
   if(MQLInfoInteger(MQL_TESTER))
      return;
   int h = FileOpen(HB_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return;
   FileWrite(h, TimeToString(TimeTradeServer(), TIME_DATE | TIME_SECONDS),
             DryRun ? "dry-run" : "live");
   FileClose(h);
  }

bool FindPosition(string sym, long magic, ulong &ticket)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) == sym &&
         PositionGetInteger(POSITION_MAGIC) == magic)
        {
         ticket = tk;
         return true;
        }
     }
   return false;
  }

bool CloseWithRetry(ulong ticket, long magic, string leg)
  {
   trade.SetExpertMagicNumber(magic);
   for(int k = 0; k < 3; k++)
     {
      if(trade.PositionClose(ticket))
         return true;
      Sleep(1000);
     }
   Log(leg, "ERROR", StringFormat("close failed: %d %s", trade.ResultRetcode(),
                                  trade.ResultRetcodeDescription()));
   return false;
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   if(MQLInfoInteger(MQL_TESTER))
      LOG_FILE = "BellCapDual_TEST_log.csv";
   trade.SetDeviationInPoints(20);
   if(EurEnabled && !SymbolSelect(EurSymbol, true))
      Log("EUR", "WARNING", "symbol not available");
   if(SpxEnabled && !SymbolSelect(SpxSymbol, true))
      Log("SPX", "WARNING", "symbol not available");
   int eur_l = (EurEntryHour - ServerMinusLocal + 24) % 24;
   int eur_x = (EurExitHour - ServerMinusLocal + 24) % 24;
   int spx_l = (SpxEntryHour - ServerMinusLocal + 24) % 24;
   int spx_x = (SpxCloseHour - ServerMinusLocal + 24) % 24;
   Log("ALL", "START", StringFormat(
          "v2.00 EUR %s %.2f lots %02d:00->%02d:00 Winnipeg | SPX %s "
          "%.2f%% swing target %02d:00->%02d:%02d Winnipeg, stop $%.0f | "
          "floor %.0f target %.0f daily %.0f",
          EurEnabled ? "on" : "off", EurBaseLots, eur_l, eur_x,
          SpxEnabled ? "on" : "off", SpxTargetVol * 100, spx_l, spx_x,
          SpxCloseMinute, SpxStopShare * InitialBalance, FloorBalance,
          TargetBalance, DailyLossLimit));
   if(eur_l != 22 || eur_x != 7)
      Log("EUR", "WARNING", "EUR times are not the tested 22:00/07:00 "
          "Winnipeg");
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) && !DryRun)
      Log("ALL", "WARNING", "Algo Trading is switched OFF in the terminal");
   EventSetTimer(10);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   Log("ALL", "STOP", StringFormat("expert removed or terminal closing (%d)",
                                   reason));
  }

void OnTick() { }
void OnTimer()
  {
   Heartbeat();
   datetime now = TimeTradeServer();
   if(EurEnabled)
      EurCheck(now);
   if(SpxEnabled)
      SpxCheck(now);
  }

bool Holiday(MqlDateTime &t)
  {
   return (t.mon == 12 && t.day == 25) || (t.mon == 1 && t.day == 1);
  }

//+------------------------------------------------------------- EUR --+
void EurCheck(datetime now)
  {
   MqlDateTime t;
   TimeToStruct(now, t);
   ulong ticket;
   if(FindPosition(EurSymbol, EurMagic, ticket))
     {
      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      if((t.hour >= EurExitHour && DayStart(opened) == DayStart(now)) ||
         DayStart(opened) < DayStart(now) || (now - opened) > 11 * 3600)
         CloseWithRetry(ticket, EurMagic, "EUR");
      return;
     }
   if(t.hour != EurEntryHour || t.min >= MaxLateMinutes)
      return;
   if(t.day_of_week < 1 || t.day_of_week > 5)
      return;
   datetime today = DayStart(now);
   if(GlobalVariableCheck(GV_EUR) &&
      (datetime)GlobalVariableGet(GV_EUR) == today)
      return;
   double bid = SymbolInfoDouble(EurSymbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(EurSymbol, SYMBOL_ASK);
   double pip = PipSize(EurSymbol);
   if((ask - bid) / pip > EurMaxSpreadPips)
     {
      if(t.min >= MaxLateMinutes - 1)
        {
         GlobalVariableSet(GV_EUR, (double)today);
         Log("EUR", "SKIP", "spread stayed too wide for the entry window");
        }
      return;
     }
   GlobalVariableSet(GV_EUR, (double)today);
   if(Holiday(t))
     {
      Log("EUR", "SKIP", "holiday");
      return;
     }
   if(AccountInfoDouble(ACCOUNT_BALANCE) >= TargetBalance)
     {
      Log("EUR", "SKIP", "target reached - trading stopped");
      return;
     }
   double share = CushionShare();
   double lots = RoundLots(EurSymbol, EurBaseLots * share);
   if(lots <= 0)
     {
      Log("EUR", "SKIP", StringFormat("size rounds to zero (allowance left "
                                      "%.0f%%)", share * 100));
      return;
     }
   double worst = (EurStopPips + SlippageAllowPips) * pip *
                  ValuePerPrice(EurSymbol) * lots + 4.0 * lots;
   double projected;
   if(!DailyRoomFor(worst, now, projected))
     {
      Log("EUR", "SKIP", StringFormat("worst case would make today's loss "
                                      "$%.2f, too close to $%.0f",
                                      projected, DailyLossLimit));
      return;
     }
   double sl = NormalizeDouble(bid + EurStopPips * pip,
                               (int)SymbolInfoInteger(EurSymbol, SYMBOL_DIGITS));
   if(DryRun)
     {
      Log("EUR", "DRY-SELL", StringFormat("would sell %.2f (allowance left "
                                          "%.0f%%), stop %.5f", lots,
                                          share * 100, sl), lots, bid, sl);
      return;
     }
   trade.SetExpertMagicNumber(EurMagic);
   trade.SetTypeFillingBySymbol(EurSymbol);
   bool ok = false;
   for(int k = 0; k < 3 && !ok; k++)
     {
      bid = SymbolInfoDouble(EurSymbol, SYMBOL_BID);
      sl = NormalizeDouble(bid + EurStopPips * pip,
                           (int)SymbolInfoInteger(EurSymbol, SYMBOL_DIGITS));
      ok = trade.Sell(lots, EurSymbol, 0, sl, 0, "BCD EUR short");
      if(!ok)
         Sleep(1000);
     }
   if(!ok)
     {
      Log("EUR", "ERROR", StringFormat("sell failed: %d %s",
                                       trade.ResultRetcode(),
                                       trade.ResultRetcodeDescription()),
          lots, bid, sl);
      return;
     }
   if(FindPosition(EurSymbol, EurMagic, ticket))
     {
      double fill = PositionGetDouble(POSITION_PRICE_OPEN);
      double want = NormalizeDouble(fill + EurStopPips * pip,
                                    (int)SymbolInfoInteger(EurSymbol,
                                                           SYMBOL_DIGITS));
      if(MathAbs(PositionGetDouble(POSITION_SL) - want) > pip / 20)
         trade.PositionModify(ticket, want, 0);
      Log("EUR", "SELL", StringFormat("sold %.2f, stop %.1f pips above fill",
                                      lots, EurStopPips), lots, fill, want);
     }
  }

//+------------------------------------------------------------- SPX --+
//--- daily (server-day) closes: returns r[1..n], r[1] = yesterday
bool SpxSignals(double &sig_today, bool &dip, int &bars_this_month,
                bool &last_day)
  {
   double c[];
   ArraySetAsSeries(c, true);
   if(CopyClose(SpxSymbol, PERIOD_D1, 1, 23, c) < 23)
      return false;
   double r[22];
   for(int k = 0; k < 22; k++)
      r[k] = c[k] / c[k + 1] - 1.0;       // r[0] = yesterday's move
   // sigma for today: std of the 20 moves up to yesterday (sample std)
   double m = 0, s = 0;
   for(int k = 0; k < 20; k++)
      m += r[k];
   m /= 20;
   for(int k = 0; k < 20; k++)
      s += (r[k] - m) * (r[k] - m);
   sig_today = MathSqrt(s / 19);
   // dip: yesterday's move against the 20 moves before it
   double m2 = 0, s2 = 0;
   for(int k = 1; k < 21; k++)
      m2 += r[k];
   m2 /= 20;
   for(int k = 1; k < 21; k++)
      s2 += (r[k] - m2) * (r[k] - m2);
   double sig_y = MathSqrt(s2 / 19);
   dip = r[0] < -SpxDipSigma * sig_y;
   // turn of month: first three trading days, or the last one
   datetime now = TimeTradeServer();
   MqlDateTime t;
   TimeToStruct(now, t);
   datetime d1[];
   ArraySetAsSeries(d1, true);
   int got = CopyTime(SpxSymbol, PERIOD_D1, 0, 30, d1);
   bars_this_month = 0;
   for(int k = 0; k < got; k++)
     {
      MqlDateTime dk;
      TimeToStruct(d1[k], dk);
      if(dk.mon == t.mon && dk.year == t.year)
         bars_this_month++;
     }
   if(got > 0)
     {
      MqlDateTime d0;
      TimeToStruct(d1[0], d0);
      if(!(d0.day == t.day && d0.mon == t.mon))
         bars_this_month++;               // today's bar not printed yet
     }
   // last trading day: the next weekday falls in another month
   datetime next = DayStart(now) + 86400;
   MqlDateTime nx;
   TimeToStruct(next, nx);
   while(nx.day_of_week == 0 || nx.day_of_week == 6)
     {
      next += 86400;
      TimeToStruct(next, nx);
     }
   last_day = (nx.mon != t.mon);
   return sig_today > 0;
  }

//--- close time today: 23:45 server, or 15 minutes before the session end
datetime SpxCloseTime(datetime now)
  {
   MqlDateTime t;
   TimeToStruct(now, t);
   datetime close_at = DayStart(now) + SpxCloseHour * 3600 +
                       SpxCloseMinute * 60;
   datetime from, to;
   for(uint s = 0; s < 4; s++)
     {
      if(!SymbolInfoSessionTrade(SpxSymbol, (ENUM_DAY_OF_WEEK)t.day_of_week,
                                 s, from, to))
         break;
      datetime end_today = DayStart(now) + (to % 86400 == 0 ? 86400 :
                                            to % 86400);
      // Only sessions that end after our entry hour matter: a short
      // session earlier in the day must not read as "closing now".
      if(end_today <= DayStart(now) + (SpxEntryHour + 1) * 3600)
         continue;
      if(end_today - 900 < close_at)
         close_at = end_today - 900;
     }
   return close_at;
  }

void SpxCheck(datetime now)
  {
   MqlDateTime t;
   TimeToStruct(now, t);
   ulong ticket;
   if(FindPosition(SpxSymbol, SpxMagic, ticket))
     {
      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      if(now >= SpxCloseTime(now) || DayStart(opened) < DayStart(now))
         CloseWithRetry(ticket, SpxMagic, "SPX");
      return;
     }
   if(t.hour != SpxEntryHour || t.min >= MaxLateMinutes)
      return;
   if(t.day_of_week < 1 || t.day_of_week > 5)
      return;
   datetime today = DayStart(now);
   if(GlobalVariableCheck(GV_SPX) &&
      (datetime)GlobalVariableGet(GV_SPX) == today)
      return;
   double bid = SymbolInfoDouble(SpxSymbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(SpxSymbol, SYMBOL_ASK);
   if(bid <= 0 || ask - bid > SpxMaxSpreadPts)
     {
      if(t.min >= MaxLateMinutes - 1)
        {
         GlobalVariableSet(GV_SPX, (double)today);
         Log("SPX", "SKIP", "market not open or spread too wide for the "
             "entry window");
        }
      return;
     }
   GlobalVariableSet(GV_SPX, (double)today);
   if(Holiday(t))
     {
      Log("SPX", "SKIP", "holiday");
      return;
     }
   if(AccountInfoDouble(ACCOUNT_BALANCE) >= TargetBalance)
     {
      Log("SPX", "SKIP", "target reached - trading stopped");
      return;
     }
   double sig;
   bool dip, last_day;
   int nth;
   if(!SpxSignals(sig, dip, nth, last_day))
     {
      Log("SPX", "SKIP", "not enough daily history for the signals");
      return;
     }
   bool tom = last_day || nth <= 3;
   double boost = (dip || tom) ? SpxBoost : 1.0;
   double share = CushionShare();
   double expo = MathMin(SpxMaxExposure, SpxTargetVol / sig) * boost * share;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double vpp = ValuePerPrice(SpxSymbol);
   double lots = RoundLots(SpxSymbol, eq * expo / (ask * vpp));
   if(lots <= 0)
     {
      Log("SPX", "SKIP", StringFormat("size rounds to zero (allowance left "
                                      "%.0f%%)", share * 100));
      return;
     }
   double stop_usd = SpxStopShare * InitialBalance;
   double stop_dist = stop_usd / (vpp * lots);
   int dg = (int)SymbolInfoInteger(SpxSymbol, SYMBOL_DIGITS);
   double sl = NormalizeDouble(ask - stop_dist, dg);
   double projected;
   if(!DailyRoomFor(stop_usd * 1.1, now, projected))
     {
      Log("SPX", "SKIP", StringFormat("worst case would make today's loss "
                                      "$%.2f, too close to $%.0f",
                                      projected, DailyLossLimit));
      return;
     }
   string why = StringFormat("sigma %.2f%%, %s%s, allowance left %.0f%%, "
                             "exposure %.2fx", sig * 100,
                             dip ? "DIP DAY " : "", tom ? "TURN OF MONTH" : "",
                             share * 100, expo);
   if(DryRun)
     {
      Log("SPX", "DRY-BUY", StringFormat("would buy %.2f; %s; stop $%.0f "
                                         "away", lots, why, stop_usd),
          lots, ask, sl);
      return;
     }
   trade.SetExpertMagicNumber(SpxMagic);
   trade.SetTypeFillingBySymbol(SpxSymbol);
   bool ok = false;
   for(int k = 0; k < 3 && !ok; k++)
     {
      ask = SymbolInfoDouble(SpxSymbol, SYMBOL_ASK);
      sl = NormalizeDouble(ask - stop_dist, dg);
      ok = trade.Buy(lots, SpxSymbol, 0, sl, 0, "BCD SPX long");
      if(!ok)
         Sleep(1000);
     }
   if(!ok)
     {
      Log("SPX", "ERROR", StringFormat("buy failed: %d %s",
                                       trade.ResultRetcode(),
                                       trade.ResultRetcodeDescription()),
          lots, ask, sl);
      return;
     }
   if(FindPosition(SpxSymbol, SpxMagic, ticket))
     {
      double fill = PositionGetDouble(POSITION_PRICE_OPEN);
      double want = NormalizeDouble(fill - stop_dist, dg);
      if(MathAbs(PositionGetDouble(POSITION_SL) - want) >
         SymbolInfoDouble(SpxSymbol, SYMBOL_POINT))
         trade.PositionModify(ticket, want, 0);
      Log("SPX", "BUY", StringFormat("bought %.2f; %s", lots, why), lots,
          fill, want);
     }
  }

//--- every closing deal of either leg, with its dollars
void OnTradeTransaction(const MqlTradeTransaction &tx,
                        const MqlTradeRequest &req,
                        const MqlTradeResult &res)
  {
   if(tx.type != TRADE_TRANSACTION_DEAL_ADD || !HistoryDealSelect(tx.deal))
      return;
   long magic = HistoryDealGetInteger(tx.deal, DEAL_MAGIC);
   string leg = magic == EurMagic ? "EUR" : (magic == SpxMagic ? "SPX" : "");
   if(leg == "")
      return;
   double comm = HistoryDealGetDouble(tx.deal, DEAL_COMMISSION);
   double vol = HistoryDealGetDouble(tx.deal, DEAL_VOLUME);
   double px = HistoryDealGetDouble(tx.deal, DEAL_PRICE);
   if(HistoryDealGetInteger(tx.deal, DEAL_ENTRY) == DEAL_ENTRY_IN)
     {
      Log(leg, "FILL-IN", "entry commission charged", vol, px, 0, 0, comm);
      return;
     }
   long why = HistoryDealGetInteger(tx.deal, DEAL_REASON);
   string how = why == DEAL_REASON_SL ? "stop hit" :
                why == DEAL_REASON_EXPERT ? "time exit" :
                "closed by hand or by the server";
   double pnl = HistoryDealGetDouble(tx.deal, DEAL_PROFIT)
                + HistoryDealGetDouble(tx.deal, DEAL_SWAP);
   Log(leg, "CLOSE", how, vol, px, 0, pnl, comm);
  }
//+------------------------------------------------------------------+
