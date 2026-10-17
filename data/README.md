# Data folder

| Folder | What | Source | Licence / note |
|---|---|---|---|
| `videos/accident/` | 12 real accident clips (2 fixed CCTV, 9 dashcam, 1 news) | GitHub: `devrishivermaa/Accident-Detection`, `mohd-faizy/realtime-accident-detection-tracking`, `Hoor-Fayaz/AI_PoweredRoadAccidentLiveCCTVApplication` (LFS) | Demo clips from public repos, academic use only, don't redistribute |
| `videos/normal/` | 4 no-event clips (3 Intel fixed-camera, 1 dashcam) for false-alarm testing | GitHub: `intel-iot-devkit/sample-videos` + Hoor-Fayaz repo | Intel clips: CC-BY 4.0 (see licence file) |
| `annotations_devset.csv` | Metadata for the 16 clips; **start/end columns still need annotating by the team** | made here | |
| `ucf_crime/Temporal_Anomaly_Annotation.txt` | Frame-level start/end labels for all 290 UCF-Crime test videos | GitHub: `WaqasSultani/AnomalyDetectionCVPR2018` | Format: `video class s1 e1 s2 e2` (frames @30 fps, -1 = none) |
| `ucf_crime/Anomaly_Train.txt` | UCF-Crime train split list (video-level labels) | same | |
| `airquality/AirQualityUCI.csv` | TD-RAG's dataset: 9,357 hourly rows, Mar 2004 – Apr 2005; `;` separator, `,` decimals, `-200` = missing | GitHub mirror of UCI ML repo #360 | CC-BY 4.0 (UCI) |

The full UCF-Crime videos and the TAD benchmark are on Google Drive / Dropbox, so run
`bash scripts/download_datasets.sh` on your own machine.
