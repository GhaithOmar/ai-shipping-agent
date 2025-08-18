# Shipping Company C — FAQ & Policies (Synthetic)
**Scope:** Heavy parcels & air cargo handoffs  
**Services:** *Air-Express Freight*, *Deferred Freight*  
**Support:** 24/7 for active air waybills

---

## Quick Facts
- **Tracking format:** `CXX-` + 8 digits (example: `CXX-44221177`)  
- **Pickup cutoff:** 12:00 local for same‑day uplift  
- **Delivery attempts:** 1 (appointment required for >30 kg)  
- **Max single piece:** 70 kg (non‑pallet), pallets by quote  
- **Default duties/taxes:** DDU; DDP by contract only

---

## Special Handling
- **Lithium batteries** only when installed in equipment and declared per IATA.  
- **Fragile/oversize** require wood‑crate or pallet; surcharge applies.  
- **Temperature‑controlled** not supported (use partner network).

---

## FAQs
### Q: Do you deliver to residential addresses?
**A:** Yes up to 30 kg. Heavier requires **dock or lift‑gate** service (book in advance).

### Q: Missed appointment?
**A:** Re‑delivery next business day; storage fees after **3 days** at destination facility.

### Q: Customs documentation?
**A:** Commercial invoice, HS codes, EORI/VAT where applicable, and any permits for restricted goods.

---

## Example Tracking
```json
{
  "tracking_id": "CXX-44221177",
  "status_code": "CLEARANCE_PROCESSING",
  "status_text": "Awaiting customs inspection",
  "events": [
    {"ts":"2025-08-16T10:11:00+03:00","code":"DEPARTED_ORIGIN","loc":"AMM-AIR"},
    {"ts":"2025-08-16T16:20:00+03:00","code":"ARRIVED_HUB","loc":"DXB-AIR"},
    {"ts":"2025-08-17T09:00:00+04:00","code":"CLEARANCE_PROCESSING","loc":"DXB-CUST"}
  ]
}
```

