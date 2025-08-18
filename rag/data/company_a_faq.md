# Shipping Company A — FAQ & Policies (Synthetic)
**Scope:** General parcel carrier (domestic, regional, international)  
**Services:** *Express*, *Economy*, *Freight-Light*  
**Support Hours:** Sun–Thu 08:00–18:00 local (emergency line 24/7)

---

## Quick Facts
- **Tracking format:** `A` + 10 digits + checksum (example: `A1234567895`)  
- **Pickup cutoff:** 14:00 local for same-day handoff  
- **Delivery attempts:** 3 (then hold at facility 7 days)  
- **Max parcel:** 30 kg (single piece), girth ≤ 300 cm, longest side ≤ 120 cm  
- **Default duties/taxes:** DDU (recipient pays), DDP optional for Express Intl.

---

## Shipment Lifecycle (high level)
1. **Label Created** → booked but not yet tendered  
2. **Picked Up** → collected from shipper  
3. **In Transit** → within hub network  
4. **Clearance Processing** (intl only)  
5. **Out for Delivery**  
6. **Delivered** *(photo/signature if required)*  
7. **Exception/Hold** *(address issue, weather, customs query, etc.)*

(See *tracking_status_reference.md* for canonical codes and synonyms.)

---

## Service SLAs (Target, Business Days)
| Service | Domestic | Regional | International |
|---|---:|---:|---:|
| Express | 1–2 | 2–3 | 3–5 |
| Economy | 2–4 | 3–5 | 5–10 |
| Freight‑Light | 3–5 | 4–7 | 7–12 |

> SLA excludes customs delays, remote areas, and force majeure.

---

## Frequently Asked Questions

### Q: How do I track a shipment?
**A:** Use your 12‑char tracking ID (e.g., `A1234567895`). Status messages include timestamps and facility codes. Example API shape for your app:
```json
{
  "tracking_id": "A1234567895",
  "status_code": "OUT_FOR_DELIVERY",
  "status_text": "Out for delivery",
  "facility": "AMM01",
  "eta": "2025-08-20",
  "events": [
    {"ts":"2025-08-18T07:02:11+03:00","code":"IN_TRANSIT","loc":"AMM01"},
    {"ts":"2025-08-18T09:20:02+03:00","code":"OUT_FOR_DELIVERY","loc":"AMM01"}
  ]
}
```

### Q: What delivery attempts are made?
**A:** Up to **3 attempts** on business days. After the 2nd failure you’ll receive an SMS/email to confirm address or **hold at location**. After 7 calendar days on hold, the parcel auto‑returns.

### Q: Can I change the delivery address?
**A:** Yes, **before** “Out for Delivery.” A re‑route may add **1 business day** and a fee.

### Q: What proof of delivery is available?
**A:** Signature capture by default for high‑value parcels (> USD 200). Otherwise geotag + photo on doorstep unless the recipient opted out.

### Q: What items are prohibited?
**A:** Aerosols, flammables, lithium batteries not in equipment, cash, live animals, human remains, perishable foods without cold chain, and items illegal at origin/destination.

### Q: How are duties/taxes handled?
**A:** Default **DDU**. For Express International, **DDP** can be enabled at booking; fees billed to shipper.

### Q: What are size/weight limits?
**A:** Single piece **≤ 30 kg**. For heavier items use **Freight‑Light** (up to 70 kg single piece, palletized).

---

## Webhooks (Push)
Send POST to your endpoint for status changes:
```json
{
  "event": "status_change",
  "tracking_id": "A1234567895",
  "previous": "IN_TRANSIT",
  "current": "OUT_FOR_DELIVERY",
  "occurred_at": "2025-08-18T09:20:02+03:00",
  "meta": {"facility":"AMM01","vehicle":"VAN-223"}
}
```

---

## Returns & Refunds
- Return-to-sender after 3 failed attempts + 7 days on hold, or on shipper request.  
- Refunds cover the **shipping fee only** when SLA missed by >1 business day (excludes customs, weather).  
- Damage claims must be filed within **7 days** of delivery with photos + packaging.

