# Feather Stories

Flag: `SIGMOID_RHGLWQ`

I filtered the observations to White Storks in Cahul and selected locality `L5819951`, which had the most ring readings (24). At each of the six positions, I took the most frequent letter after excluding `?`. The consensus, `RHGLWQ`, appears verbatim five times and is absent from the ringing register.

Download the two CSVs from the [task data folder](https://drive.google.com/drive/folders/1R_cYNcO-t0MRR2ri6Nhgj1cSJ7zCiWBM), then run this command from the repository root.

```powershell
./feather_stories/solve.ps1 -DataDirectory 'path/to/csv-folder'
```

## Submission status and cross-check

The platform returned 0 points for this flag. It gave no specific reason. Treat the flag above as a candidate supported by the supplied files.

Run `./feather_stories/audit.ps1 -DataDirectory 'path/to/csv-folder'` to compare localities with at least five readings. The script reports consensus codes that occur verbatim and are absent from the register. The five exact readings of RHGLWQ came from five distinct observer groups. The other two consensus codes with five exact readings were IYRNDQ (Căușeni, two observer groups) and EKBDAO (Șoldănești, one observer group).
