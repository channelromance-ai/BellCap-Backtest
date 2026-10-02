//+------------------------------------------------------------------+
//| DumpBars.mq5                                                       |
//| Export broker history for the BellCap quant battery.               |
//|                                                                    |
//| For every symbol (Market Watch, or the list you give) it writes   |
//| to MQL5\Files\bellcap\ :                                           |
//|   <SYM>_M1.csv     1-minute bid bars: time, open, high, low,       |
//|                    close, tick_volume, spread (points)             |
//|   <SYM>_ticks.csv  optional: per-minute bid/ask statistics built   |
//|                    from real ticks: n_ticks, mean and max spread,  |
//|                    last bid and ask (points / prices)              |
//|   symbols.csv      contract specs, swaps, and the server's offset  |
//|                    from GMT, so costs and times can be rebuilt     |
//|                                                                    |
//| Times are broker server time, as integers (seconds since 1970).   |
//| Run it as a script: drag onto any chart, set the inputs, OK.       |
//| History is pulled month by month, so years of data fit in memory. |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input string   InpSymbols  = "";             // Symbols, comma separated (blank = Market Watch)
input datetime InpFrom     = D'2005.01.01';  // From (server time)
input datetime InpTo       = D'2030.01.01';  // To (server time; capped at now)
input bool     InpBars     = true;           // Export 1-minute bars
input bool     InpTicks    = true;           // Export per-minute tick spread stats
input datetime InpTicksFrom= D'2020.01.01';  // Ticks from (tick history is short and big)

string DIR = "bellcap\\";

//+------------------------------------------------------------------+
int OnStart()
  {
   string syms[];
   int n = 0;
   if(StringLen(InpSymbols) > 0)
      n = StringSplit(InpSymbols, ',', syms);
   else
     {
      n = SymbolsTotal(true);
      ArrayResize(syms, n);
      for(int i = 0; i < n; i++)
         syms[i] = SymbolName(i, true);
     }
   datetime to = MathMin(InpTo, TimeCurrent());
   WriteSpecs(syms, n);
   for(int i = 0; i < n; i++)
     {
      string s = syms[i];
      StringTrimLeft(s);
      StringTrimRight(s);
      if(!SymbolSelect(s, true))
        {
         PrintFormat("skip %s: not available", s);
         continue;
        }
      if(InpBars)
         DumpBars(s, InpFrom, to);
      if(InpTicks)
         DumpTicks(s, MathMax(InpTicksFrom, InpFrom), to);
      if(IsStopped())
         break;
     }
   Print("DumpBars finished. Files are in MQL5\\Files\\", DIR);
   return 0;
  }

//+------------------------------------------------------------------+
datetime NextMonth(datetime t)
  {
   MqlDateTime d;
   TimeToStruct(t, d);
   d.day = 1; d.hour = 0; d.min = 0; d.sec = 0;
   d.mon++;
   if(d.mon > 12) { d.mon = 1; d.year++; }
   return StructToTime(d);
  }

//+------------------------------------------------------------------+
// CopyRates can return -1 while the terminal is still downloading
// history; wait and retry a few times before giving up on a chunk.
int RatesWithRetry(string s, datetime a, datetime b, MqlRates &r[])
  {
   for(int k = 0; k < 20; k++)
     {
      ResetLastError();
      int got = CopyRates(s, PERIOD_M1, a, b, r);
      if(got >= 0)
         return got;
      Sleep(500);
     }
   return -1;
  }

//+------------------------------------------------------------------+
void DumpBars(string s, datetime from, datetime to)
  {
   int h = FileOpen(DIR + s + "_M1.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE)
     {
      PrintFormat("cannot open file for %s: %d", s, GetLastError());
      return;
     }
   FileWrite(h, "time", "open", "high", "low", "close", "tick_volume", "spread");
   long rows = 0;
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   for(datetime a = from; a < to && !IsStopped(); a = NextMonth(a))
     {
      datetime b = MathMin(NextMonth(a) - 1, to);
      MqlRates r[];
      int got = RatesWithRetry(s, a, b, r);
      if(got <= 0)
         continue;
      for(int j = 0; j < got; j++)
         FileWrite(h, (long)r[j].time,
                   DoubleToString(r[j].open, digits),
                   DoubleToString(r[j].high, digits),
                   DoubleToString(r[j].low, digits),
                   DoubleToString(r[j].close, digits),
                   r[j].tick_volume, r[j].spread);
      rows += got;
     }
   FileClose(h);
   PrintFormat("%s: %I64d one-minute bars", s, rows);
  }

//+------------------------------------------------------------------+
// One day of ticks at a time, folded into one row per minute with the
// real quoted spread (mean and max), so the cost of trading at any
// minute -- news, the rollover hour -- is known rather than assumed.
void DumpTicks(string s, datetime from, datetime to)
  {
   int h = FileOpen(DIR + s + "_ticks.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE)
     {
      PrintFormat("cannot open tick file for %s: %d", s, GetLastError());
      return;
     }
   FileWrite(h, "minute", "n_ticks", "spread_mean", "spread_max", "bid", "ask");
   double pt = SymbolInfoDouble(s, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(s, SYMBOL_DIGITS);
   long rows = 0;
   for(datetime a = from; a < to && !IsStopped(); a += 86400)
     {
      MqlTick t[];
      int got = -1;
      for(int k = 0; k < 20 && got < 0; k++)
        {
         got = CopyTicksRange(s, t, COPY_TICKS_INFO,
                              (ulong)a * 1000, (ulong)(a + 86400) * 1000 - 1);
         if(got < 0)
            Sleep(500);
        }
      if(got <= 0)
         continue;
      long cur = -1;
      int cnt = 0;
      double sum = 0, mx = 0, bid = 0, ask = 0;
      for(int j = 0; j <= got; j++)
        {
         long m = (j < got) ? (long)(t[j].time / 60) * 60 : -2;
         if(m != cur && cur >= 0 && cnt > 0)
           {
            FileWrite(h, cur, cnt, DoubleToString(sum / cnt, 2),
                      DoubleToString(mx, 1), DoubleToString(bid, digits),
                      DoubleToString(ask, digits));
            rows++;
            cnt = 0; sum = 0; mx = 0;
           }
         if(j == got)
            break;
         cur = m;
         if(t[j].bid <= 0 || t[j].ask <= 0)
            continue;
         double sp = (t[j].ask - t[j].bid) / pt;
         cnt++;
         sum += sp;
         mx = MathMax(mx, sp);
         bid = t[j].bid;
         ask = t[j].ask;
        }
     }
   FileClose(h);
   PrintFormat("%s: %I64d minutes of tick spread", s, rows);
  }

//+------------------------------------------------------------------+
void WriteSpecs(string &syms[], int n)
  {
   int h = FileOpen(DIR + "symbols.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE)
      return;
   FileWrite(h, "sym", "digits", "point", "contract_size", "tick_value",
             "tick_size", "swap_long", "swap_short", "swap_mode",
             "swap_3day", "profit_ccy", "margin_ccy", "description",
             "server_minus_gmt_sec");
   long off = (long)(TimeTradeServer() - TimeGMT());
   for(int i = 0; i < n; i++)
     {
      string s = syms[i];
      StringTrimLeft(s);
      StringTrimRight(s);
      if(!SymbolSelect(s, true))
         continue;
      FileWrite(h, s,
                SymbolInfoInteger(s, SYMBOL_DIGITS),
                SymbolInfoDouble(s, SYMBOL_POINT),
                SymbolInfoDouble(s, SYMBOL_TRADE_CONTRACT_SIZE),
                SymbolInfoDouble(s, SYMBOL_TRADE_TICK_VALUE),
                SymbolInfoDouble(s, SYMBOL_TRADE_TICK_SIZE),
                SymbolInfoDouble(s, SYMBOL_SWAP_LONG),
                SymbolInfoDouble(s, SYMBOL_SWAP_SHORT),
                SymbolInfoInteger(s, SYMBOL_SWAP_MODE),
                SymbolInfoInteger(s, SYMBOL_SWAP_ROLLOVER3DAYS),
                SymbolInfoString(s, SYMBOL_CURRENCY_PROFIT),
                SymbolInfoString(s, SYMBOL_CURRENCY_MARGIN),
                SymbolInfoString(s, SYMBOL_DESCRIPTION),
                off);
     }
   FileClose(h);
  }
//+------------------------------------------------------------------+
