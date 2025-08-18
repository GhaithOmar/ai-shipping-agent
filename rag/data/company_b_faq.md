# Shipping Company B — FAQ & Policies (Synthetic)
**Scope:** Cross‑border B2C parcels (marketplaces/SMBs)  
**Services:** *Next‑Day Domestic*, *Saver Regional*, *Cross‑Border Economy*  
**Support:** Mon–Sat 09:00–19:00 local

---

## Quick Facts
- **Tracking format:** 12 digits numeric (example: `823456789012`)  
- **Pickup cutoff:** 16:00 local (merchant hubs until 18:00)  
- **Delivery attempts:** 2 (then hold 5 days)  
- **Max parcel:** 25 kg, longest side ≤ 100 cm  
- **Default duties/taxes:** **DDP** for B2C shipments (taxes pre‑collected)

---

## Service SLAs (Target, Business Days)
| Service | Domestic | Regional | International |
|---|---:|---:|---:|
| Next‑Day | 1 | — | — |
| Saver | 2–3 | 3–4 | — |
| Cross‑Border Economy | — | 4–7 | 6–12 |

---

## FAQs

### Q: Can recipients schedule a time window?
**A:** Yes—two‑hour windows in major cities. Remote areas get a day‑level ETA.

### Q: What payment options exist at delivery?
**A:** **COD** (cash/card) optional; fee applies. COD allowed up to USD 500 equivalent.

### Q: Address correction?
**A:** Corrections accepted until **Out for Delivery**. After that, you can **reschedule next day**.

### Q: DDP vs DDU?
**A:** B2C defaults to **DDP** (we remit taxes). Business‑to‑business may choose DDU.

### Q: Prohibited items?
**A:** Same baseline as Company A, plus liquids over 100 ml and cosmetic aerosols unless fully compliant.

---

## Example Tracking (normalized)
```json
{
  "tracking_id": "823456789012",
  "status_code": "DELIVERED",
  "status_text": "Delivered (signature: H. Omar)",
  "eta": "2025-08-19",
  "events": [
    {"ts":"2025-08-17T13:04:00+03:00","code":"PICKED_UP","loc":"AMM-HUB"},
    {"ts":"2025-08-18T09:08:00+03:00","code":"OUT_FOR_DELIVERY","loc":"AMM-DC"},
    {"ts":"2025-08-18T16:41:00+03:00","code":"DELIVERED","loc":"AMM-DC","pod":"H. Omar"}
  ]
}
```

---

## Returns
- 2 attempts; then 5‑day hold; then **auto‑return**.  
- Unpaid duties on DDU shipments are billed to shipper upon return.

