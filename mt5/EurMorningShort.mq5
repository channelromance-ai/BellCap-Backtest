//+------------------------------------------------------------------+
//| EurMorningShort.mq5                                              |
//|                                                                  |
//| The European-morning EUR/USD short, exactly as tested in         |
//| examples/eur_morning_late.py:                                    |
//|                                                                  |
//|   SELL EURUSD at 22:00 Winnipeg (05:00 server), Sun-Thu nights   |
//|   fixed lots, stop 40 pips above the fill, no target             |
//|   CLOSE at 07:00 Winnipeg (15:00 server), whatever it is doing   |
//|                                                                  |
//| The5ers' server runs on New York time + 7 hours all year, and    |
//| Winnipeg is New York - 1, so both times are fixed on the server  |
//| clock and daylight saving never moves them.                      |
//|                                                                  |
//| It only ever touches positions carrying its own magic number.    |
//| Every action, every skipped night and every dollar is written to |
//| a CSV in the terminal's Common\Files folder.                     |
//+------------------------------------------------------------------+
#property copyright "BellCap-Backtest"
#property version   "1.00"

#include <Trade\Trade.mqh>

//--- inputs
input bool   DryRun            = true;     // true = log only, place no trades
input string TradeSymbol       = "EURUSD";
input double Lots              = 0.18;     // fixed size
input double StopPips          = 40.0;     // stop distance above the fill
input int    EntryHourServer   = 5;        // 05:00 server = 22:00 Winnipeg
input int    ExitHourServer    = 15;       // 15:00 server = 07:00 Winnipeg
input int    MaxLateMinutes    = 15;       // never enter later than this
input double MaxSpreadPips     = 2.0;      // skip the night if wider
input double TargetBalance     = 5500.0;   // stop trading once reached
input double FloorBalance      = 4700.0;   // The5ers overall loss level
input double DailyLossLimit    = 150.0;    // The5ers daily loss, in $
input double SlippageAllowPips = 2.0;      // assumed extra loss on a stop
input int    ServerMinusLocal  = 8;        // server hours ahead of Winnipeg
input long   Magic             = 26092501;

CTrade trade;
string LOG_FILE = "EurMorningShort_log.csv";
string GV_LAST  = "EMS_last_entry_day";

//+------------------------------------------------------------------+
double PipSize()
  {
   int digits = (int)SymbolInfoInteger(TradeSymbol, SYMBOL_DIGITS);
   return (digits == 3 || digits == 5) ? 10 * _Point : _Point;
  }

//--- dollars lost if price moves `pips` against `lots`
double DollarsFor(double pips, double lots)
  {
   double tv = SymbolInfoDouble(TradeSymbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(TradeSymbol, SYMBOL_TRADE_TICK_SIZE);
   if(ts <= 0)
      return 0;
   return pips * PipSize() / ts * tv * lots;
  }

//--- start of the current server day (The5ers' daily reset)
datetime DayStart(datetime t)
  {
   MqlDateTime d;
   TimeToStruct(t, d);
   d.hour = 0; d.min = 0; d.sec = 0;
   return StructToTime(d);
  }

//--- closed P&L since the server day began, all symbols, all sources
double ClosedToday(datetime now)
  {
   double sum = 0;
   if(!HistorySelect(DayStart(now), now + 60))
      return 0;
   int n = HistoryDealsTotal();
   for(int i = 0; i < n; i++)
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

//+------------------------------------------------------------------+
void Log(string event, string detail, double lots = 0, double price = 0,
         double sl = 0, double pnl = 0, double comm = 0)
  {
   datetime srv = TimeTradeServer();
   datetime loc = srv - ServerMinusLocal * 3600;
   int h = FileOpen(LOG_FILE, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI |
                    FILE_COMMON | FILE_SHARE_READ, ',');
   if(h == INVALID_HANDLE)
     {
      Print("EMS log open failed: ", GetLastError());
      return;
     }
   if(FileSize(h) == 0)
      FileWrite(h, "server_time", "winnipeg_time", "event", "detail", "lots",
                "price", "stop", "pnl", "commission", "balance", "equity",
                "dry_run");
   FileSeek(h, 0, SEEK_END);
   FileWrite(h, TimeToString(srv, TIME_DATE | TIME_SECONDS),
             TimeToString(loc, TIME_DATE | TIME_SECONDS), event, detail,
             DoubleToString(lots, 2), DoubleToString(price, 5),
             DoubleToString(sl, 5), DoubleToString(pnl, 2),
             DoubleToString(comm, 2),
             DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
             DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2),
             DryRun ? "yes" : "no");
   FileClose(h);
   Print("EMS ", event, ": ", detail);
  }

//--- our open position, if any
bool OurPosition(ulong &ticket)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong tk = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == TradeSymbol &&
         PositionGetInteger(POSITION_MAGIC) == Magic)
        {
         ticket = tk;
         return true;
        }
     }
   return false;
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   // A Strategy Tester run must never write into the real trading record.
   if(MQLInfoInteger(MQL_TESTER))
      LOG_FILE = "EurMorningShort_TEST_log.csv";
   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(20);
   trade.SetTypeFillingBySymbol(TradeSymbol);
   if(!SymbolSelect(TradeSymbol, true))
     {
      Print("EMS: symbol not available");
      return INIT_FAILED;
     }
   double risk = DollarsFor(StopPips, Lots);
   Log("START", StringFormat("lots %.2f stop %.0f pips = $%.2f at risk; "
                             "entry %02d:00 server, exit %02d:00 server",
                             Lots, StopPips, risk, EntryHourServer,
                             ExitHourServer), Lots);
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) && !DryRun)
      Log("WARNING", "Algo Trading is switched OFF in the terminal - "
          "no trades will be placed until it is switched on");
   EventSetTimer(10);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   Log("STOP", StringFormat("expert removed or terminal closing (%d)",
                            reason));
  }

void OnTick() { Check(); }
void OnTimer() { Check(); }

//+------------------------------------------------------------------+
void Check()
  {
   datetime now = TimeTradeServer();
   MqlDateTime t;
   TimeToStruct(now, t);
   ulong ticket;
   bool have = OurPosition(ticket);

   //--- EXIT: at or after the exit hour, or anything older than 11 hours
   if(have)
     {
      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      bool exit_time = (t.hour >= ExitHourServer &&
                        DayStart(opened) == DayStart(now)) ||
                       DayStart(opened) < DayStart(now) ||
                       (now - opened) > 11 * 3600;
      if(exit_time)
        {
         double px = PositionGetDouble(POSITION_PRICE_CURRENT);
         double vol = PositionGetDouble(POSITION_VOLUME);
         bool ok = false;
         for(int k = 0; k < 3 && !ok; k++)
           {
            ok = trade.PositionClose(ticket);
            if(!ok)
               Sleep(1000);
           }
         if(!ok)
            Log("ERROR", StringFormat("time exit failed: %d %s",
                                      trade.ResultRetcode(),
                                      trade.ResultRetcodeDescription()),
                vol, px);
         // the P&L itself is logged by OnTradeTransaction
        }
      return;
     }

   //--- ENTRY window only
   if(t.hour != EntryHourServer || t.min >= MaxLateMinutes)
      return;
   if(t.day_of_week < 1 || t.day_of_week > 5)
      return;
   datetime today = DayStart(now);
   if(GlobalVariableCheck(GV_LAST) &&
      (datetime)GlobalVariableGet(GV_LAST) == today)
      return;                                    // already handled today

   // A wide spread can be a passing blip, so keep checking until the
   // window closes; log it once rather than every ten seconds.
   static datetime spread_logged = 0;
   double bid0 = SymbolInfoDouble(TradeSymbol, SYMBOL_BID);
   double ask0 = SymbolInfoDouble(TradeSymbol, SYMBOL_ASK);
   double spread0 = (ask0 - bid0) / PipSize();
   if(spread0 > MaxSpreadPips)
     {
      if(spread_logged != today)
        {
         Log("WAIT", StringFormat("spread %.1f pips is above %.1f - will "
                                  "keep checking until %02d:%02d server",
                                  spread0, MaxSpreadPips, EntryHourServer,
                                  MaxLateMinutes), 0, bid0);
         spread_logged = today;
        }
      if(t.min >= MaxLateMinutes - 1)
        {
         GlobalVariableSet(GV_LAST, (double)today);
         Log("SKIP", "spread stayed too wide for the whole entry window");
        }
      return;
     }
   GlobalVariableSet(GV_LAST, (double)today);

   if((t.mon == 12 && t.day == 25) || (t.mon == 1 && t.day == 1))
     {
      Log("SKIP", "holiday: thin market");
      return;
     }
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(bal >= TargetBalance)
     {
      Log("SKIP", StringFormat("target reached (balance %.2f) - trading "
                               "stopped", bal));
      return;
     }
   double bid = SymbolInfoDouble(TradeSymbol, SYMBOL_BID);
   // Worst realistic loss: the stop plus slippage plus commission. If that
   // would breach The5ers' daily limit, do not trade.
   double worst = DollarsFor(StopPips + SlippageAllowPips, Lots) + 4.0 * Lots;
   double start_bal = bal - ClosedToday(now);
   double day_loss_if_stopped = start_bal - (eq - worst);
   if(day_loss_if_stopped >= DailyLossLimit * 0.95)
     {
      Log("SKIP", StringFormat("a stop-out would make today's loss $%.2f, "
                               "too close to the $%.0f daily limit",
                               day_loss_if_stopped, DailyLossLimit));
      return;
     }
   if(eq - worst <= FloorBalance)
      Log("WARNING", StringFormat("a stop-out tonight would take equity to "
                                  "about $%.2f, at or below the $%.0f floor",
                                  eq - worst, FloorBalance));

   double sl = NormalizeDouble(bid + StopPips * PipSize(), _Digits);
   if(DryRun)
     {
      Log("DRY-SELL", StringFormat("would sell %.2f at %.5f, stop %.5f, "
                                   "$%.2f at risk", Lots, bid, sl,
                                   DollarsFor(StopPips, Lots)),
          Lots, bid, sl);
      return;
     }

   bool ok = false;
   for(int k = 0; k < 3 && !ok; k++)
     {
      bid = SymbolInfoDouble(TradeSymbol, SYMBOL_BID);
      sl = NormalizeDouble(bid + StopPips * PipSize(), _Digits);
      ok = trade.Sell(Lots, TradeSymbol, 0, sl, 0, "EMS 22:00 short");
      if(!ok)
         Sleep(1000);
     }
   if(!ok)
     {
      Log("ERROR", StringFormat("sell failed: %d %s", trade.ResultRetcode(),
                                trade.ResultRetcodeDescription()),
          Lots, bid, sl);
      return;
     }
   // Put the stop exactly StopPips above the actual fill.
   if(OurPosition(ticket))
     {
      double fill = PositionGetDouble(POSITION_PRICE_OPEN);
      double want = NormalizeDouble(fill + StopPips * PipSize(), _Digits);
      if(MathAbs(PositionGetDouble(POSITION_SL) - want) > _Point / 2)
         trade.PositionModify(ticket, want, 0);
      Log("SELL", StringFormat("sold %.2f, stop %.1f pips above the fill",
                               Lots, StopPips), Lots, fill, want);
     }
  }

//--- log every closing deal of ours, stop or time exit, with its dollars
void OnTradeTransaction(const MqlTradeTransaction &tx,
                        const MqlTradeRequest &req,
                        const MqlTradeResult &res)
  {
   if(tx.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   if(!HistoryDealSelect(tx.deal))
      return;
   if(HistoryDealGetInteger(tx.deal, DEAL_MAGIC) != Magic)
      return;
   long entry = HistoryDealGetInteger(tx.deal, DEAL_ENTRY);
   double comm = HistoryDealGetDouble(tx.deal, DEAL_COMMISSION);
   double vol = HistoryDealGetDouble(tx.deal, DEAL_VOLUME);
   double px = HistoryDealGetDouble(tx.deal, DEAL_PRICE);
   if(entry == DEAL_ENTRY_IN)
     {
      Log("FILL-IN", "entry commission charged", vol, px, 0, 0, comm);
      return;
     }
   long why = HistoryDealGetInteger(tx.deal, DEAL_REASON);
   string how = (why == DEAL_REASON_SL) ? "stop hit" :
                (why == DEAL_REASON_EXPERT) ? "07:00 time exit" :
                "closed by hand or by the server";
   double pnl = HistoryDealGetDouble(tx.deal, DEAL_PROFIT)
                + HistoryDealGetDouble(tx.deal, DEAL_SWAP);
   Log("CLOSE", how, vol, px, 0, pnl, comm);
  }
//+------------------------------------------------------------------+
