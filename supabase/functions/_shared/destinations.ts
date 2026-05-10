// Mirror of DESTINATIONS in app.py. Kept in sync manually — destinations rarely
// change. If they do, update both this file and app.py:105 in the same commit.

export const DESTINATIONS = {
  beachfront: [
    { name: "Phuket", airport: "HKT", flight_usd: 150 },
    { name: "Khao Lak", airport: "HKT", flight_usd: 150 },
    { name: "Koh Samui", airport: "USM", flight_usd: 280 },
    { name: "Da Nang", airport: "DAD", flight_usd: 280 },
    { name: "Phu Quoc", airport: "PQC", flight_usd: 250 },
    { name: "Bali", airport: "DPS", flight_usd: 128 },
    { name: "Lombok", airport: "LOP", flight_usd: 185 },
    { name: "Langkawi", airport: "LGK", flight_usd: 137 },
    { name: "Sihanoukville", airport: "KOS", flight_usd: 200 },
    { name: "Bentota", airport: "CMB", flight_usd: 253 },
    { name: "Nha Trang", airport: "CXR", flight_usd: 282 },
    { name: "Hoi An", airport: "DAD", flight_usd: 280 },
    { name: "Bintan", airport: "TNJ", flight_usd: 60 },
  ],
  non_beachfront: [
    { name: "Kuala Lumpur", airport: "KUL", flight_usd: 77 },
    { name: "Bangkok", airport: "BKK", flight_usd: 126 },
    { name: "Jakarta", airport: "CGK", flight_usd: 107 },
    { name: "Ho Chi Minh City", airport: "SGN", flight_usd: 113 },
    { name: "Hanoi", airport: "HAN", flight_usd: 226 },
    { name: "Colombo", airport: "CMB", flight_usd: 253 },
    { name: "Siem Reap", airport: "SAI", flight_usd: 188 },
    { name: "Phnom Penh", airport: "PNH", flight_usd: 200 },
    { name: "Yogyakarta", airport: "JOG", flight_usd: 150 },
  ],
};
