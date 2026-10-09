# BellCapDual watchdog, run daily at 16:40 and 21:40 Winnipeg, and one minute
# after logging on to Windows, by Task Scheduler
#
# MT5 runs on a VPS, so this script no longer starts MT5 on this PC: doing
# so loaded a second, live copy of BellCapDual here (9 Oct 2026, 14:11),
# which could double every entry while the VPS copy is also trading.
#
# 1. Check the expert's heartbeat file. If it is missing, more than 3
#    minutes old, or the expert is in dry-run mode, pop up a warning so
#    there is still time before the next entry (S&P 17:00, EUR 22:00).
# 2. Same age check for the XAU intraday expert (paper mode), once it has
#    written a heartbeat at least once. Until then it is skipped.
#
# The heartbeat files are read from THIS computer's MT5 folder; they only
# reflect the VPS if this script runs on the VPS itself.
#
# Remove with:  Unregister-ScheduledTask -TaskName "BellCapDual watchdog"

$heartbeat = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\BellCapDual_heartbeat.txt"
$log       = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\BellCapDual_watchdog.log"
$xauBeat   = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\XAU_heartbeat.txt"

function Note($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content $log }

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

if (Test-Path $xauBeat) {
    $age = (Get-Date) - (Get-Item $xauBeat).LastWriteTime
    if ($age.TotalMinutes -gt 3) {
        $xau = "XAU intraday (paper) is NOT running (last heartbeat $([int]$age.TotalMinutes) min ago). Attach it to an XAUUSD chart."
        if ($problem) { $problem = "$problem`n`n$xau" } else { $problem = $xau }
    }
}

if ($problem) {
    Note "ALERT: $problem"
    $shell = New-Object -ComObject WScript.Shell
    $null = $shell.Popup($problem, 0, "MT5 experts - action needed", 0x30)
} else {
    Note "OK - expert alive and live"
}
