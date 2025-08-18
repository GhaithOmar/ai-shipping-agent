# Tracking Status Reference (Synthetic Canonical Dictionary)

Use these canonical codes in your system and map carrier‑specific strings to them for a consistent UX.

| Canonical Code | Common Label | Example Synonyms |
|---|---|---|
| LABEL_CREATED | Label created | Shipment info sent, Pre‑advice received |
| PICKED_UP | Picked up | Collected, Tendered |
| IN_TRANSIT | In transit | Line‑haul, Arrived hub, Departed facility |
| CLEARANCE_PROCESSING | Clearance processing | Customs hold, Awaiting inspection |
| OUT_FOR_DELIVERY | Out for delivery | With courier, On vehicle for delivery |
| DELIVERED | Delivered | Signed for, Proof of delivery |
| EXCEPTION | Exception | Address issue, Weather delay, Security delay |
| HELD_AT_LOCATION | Held at location | Customer requested hold, Pickup at station |
| RETURN_TO_SENDER | Return to sender | RTS initiated, Returned |
| CANCELLED | Cancelled | Void label |

**Notes**
- Show both the **canonical** label and the **carrier raw text** for transparency.  
- Keep a **synonym list** in your embeddings to boost recall for user queries such as *"where is my parcel"*, *"with the courier"*, *"stuck in customs"*, etc.
