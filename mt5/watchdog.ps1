# BellCapDual watchdog, run daily at 16:40 and 21:40 Winnipeg by Task Scheduler
#
# 1. If MetaTrader 5 is not running, start it (it reloads the charts and
#    experts it had open when it last closed normally).
# 2. Two minutes later, check the expert's heartbeat file. If it is missing,
#    more than 3 minutes old, or the expert is in dry-run mode, pop up a
#    warning so there is still time before the next entry (S&P 17:00, EUR 22:00).
#
# Remove with:  Unregister-ScheduledTask -TaskName "BellCapDual watchdog"

$terminal  = "C:\Program Files\MetaTrader 5\terminal64.exe"
$heartbeat = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\BellCapDual_heartbeat.txt"
$log       = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\BellCapDual_watchdog.log"

function Note($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content $log }

if (-not (Get-Process terminal64 -ErrorAction SilentlyContinue)) {
    Start-Process $terminal
    Note "MT5 was not running - started it"
}
Start-Sleep -Seconds 120

$problem = $null
if (-not (Test-Path $heartbeat)) {
    $problem = "BellCapDual has never written its heartbeat. Attach it to a chart."
} else {
    $age = (Get-Date) - (Get-Item $heartbeat).LastWriteTime
    $mode = (Get-Content $heartbeat -Raw)
    if ($age.TotalMinutes -gt 3) {
        $problem = "BellCapDual is NOT running (last heartbeat $([int]$age.TotalMinutes) min ago). Attach it before the next entry (S&P 5:00 pm, EUR 10:00 pm)."
    } elseif ($mode -match "dry-run") {
        $problem = "BellCapDual is running in DRY-RUN mode and will not trade. Set DryRun = false before the next entry (S&P 5:00 pm, EUR 10:00 pm)."
    }
}

if ($problem) {
    Note "ALERT: $problem"
    $shell = New-Object -ComObject WScript.Shell
    $null = $shell.Popup($problem, 0, "BellCapDual - action needed", 0x30)
} else {
    Note "OK - expert alive and live"
}
