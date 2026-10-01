# EurMorningShort watchdog, run daily at 21:40 Winnipeg by Task Scheduler.
#
# 1. If MetaTrader 5 is not running, start it (it reloads the charts and
#    experts it had open when it last closed normally).
# 2. Two minutes later, check the expert's heartbeat file. If it is missing,
#    more than 3 minutes old, or the expert is in dry-run mode, pop up a
#    warning so there is still time to attach it before the 22:00 entry.
#
# Remove with:  Unregister-ScheduledTask -TaskName "EurMorningShort watchdog"

$terminal  = "C:\Program Files\MetaTrader 5\terminal64.exe"
$heartbeat = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\EurMorningShort_heartbeat.txt"
$log       = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\EurMorningShort_watchdog.log"

function Note($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content $log }

if (-not (Get-Process terminal64 -ErrorAction SilentlyContinue)) {
    Start-Process $terminal
    Note "MT5 was not running - started it"
}
Start-Sleep -Seconds 120

$problem = $null
if (-not (Test-Path $heartbeat)) {
    $problem = "EurMorningShort has never written its heartbeat. Attach it to a EURUSD chart."
} else {
    $age = (Get-Date) - (Get-Item $heartbeat).LastWriteTime
    $mode = (Get-Content $heartbeat -Raw)
    if ($age.TotalMinutes -gt 3) {
        $problem = "EurMorningShort is NOT running (last heartbeat $([int]$age.TotalMinutes) min ago). Attach it to a EURUSD chart before 10:00 pm."
    } elseif ($mode -match "dry-run") {
        $problem = "EurMorningShort is running in DRY-RUN mode and will not trade. Set DryRun = false before 10:00 pm."
    }
}

if ($problem) {
    Note "ALERT: $problem"
    $shell = New-Object -ComObject WScript.Shell
    $null = $shell.Popup($problem, 0, "EUR/USD short - action needed", 0x30)
} else {
    Note "OK - expert alive and live"
}
