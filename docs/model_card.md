# Model card - OCAT-Net-S reconstruction
* Status: no trained weights released; none evaluated on real data.
* Intended use: research on image-level Normal vs Oral Cancer classification from RGB oral photographs. Not for clinical use or autonomous decisions.
* Architecture: conv stem, MBConv, MBConv-SE, CPE, TFSG (DCT-gated windowed attention); 4.73 M parameters as specified literally (paper: 3.55 M).
* Output: Oral-Cancer probability after validation-only temperature scaling and a validation-selected threshold.
* Limits: image-level labels only; filename-proxy groups are not patients; no demographic/acquisition metadata, so subgroup bias cannot be assessed; external transfer in the paper dropped to AUC .83-.84; Grad-CAM is qualitative (no lesion masks).
* Risks: false negatives may delay referral; false positives create anxiety and workload. Any use requires clinician oversight and prospective, patient-level, multi-site validation.
