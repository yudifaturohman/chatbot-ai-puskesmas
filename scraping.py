import time
import csv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# List URL puskesmas
urls = [
    'https://pkmbandung.serangkab.go.id/page/visi-misi',
    'https://pkmanyar.serangkab.go.id/page/visi-misi',
    'https://pkmciomas.serangkab.go.id/page/visi-misi',
    'https://pkmbaros.serangkab.go.id/page/visi-misi-dan-tujuan',
    'https://pkmbinuang.serangkab.go.id/page/visi-misi',
    'https://pkmbojonegara.serangkab.go.id/page/visi-misi',
    'https://pkmcarenang.serangkab.go.id/page/visi-misi',
    'https://pkmcikande.serangkab.go.id/page/visi-misi',
    'https://pkmcikeusal.serangkab.go.id/page/visi-misi',
    'https://pkmcinangka.serangkab.go.id/page/visi-misi',
    'https://pkmciomas.serangkab.go.id/page/visi-misi',
    'https://pkmciruas.serangkab.go.id/page/visi-misi',
    'https://pkmgunungsari.serangkab.go.id/page/visi-misi',
    'https://pkmjawilan.serangkab.go.id/page/visi-misi',
    'https://pkmkibin.serangkab.go.id/page/visi-misi',
    'https://pkmkopo.serangkab.go.id/page/visi-misi',
    'https://pkmkragilan.serangkab.go.id/page/visi-misi',
    'https://pkmkramatwatu.serangkab.go.id/page/visi-misi',
    'https://pkmlebakwangi.serangkab.go.id/page/visi-misi',
    'https://pkmmancak.serangkab.go.id/page/visi-misi',
    'https://pkmnyompok.serangkab.go.id/page/visi-misi',
    'https://pkmpabuaran.serangkab.go.id/page/visi-misi',
    'https://pkmpadarincang.serangkab.go.id/page/visi-misi',
    'https://pkmpamarayan.serangkab.go.id/page/visi-misi',
    'https://pkmpematang.serangkab.go.id/page/visi-misi',
    'https://pkmpetir.serangkab.go.id/page/visi-misi',
    'https://pkmpontang.serangkab.go.id/page/visi-misi',
    'https://pkmpuloampel.serangkab.go.id/page/visi-misi',
    'https://pkmtanara.serangkab.go.id/page/visi-misi',
    'https://pkmtirtayasa.serangkab.go.id/page/visi-misi',
    'https://pkmtunjungteja.serangkab.go.id/page/visi-misi',
    'https://pkmwaringinkurung.serangkab.go.id/page/visi-misi-motto-dan-tata-nilai-upt-puskesmas-waringinkurung',
]

# Setup browser (headless Chrome)
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# List hasil akhir
semua_data = []

# Loop semua URL
for url in urls:
    print(f"🔄 Memproses: {url}")
    try:
        driver.get(url)
        time.sleep(5)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Ambil nama puskesmas dari tag <title>
        title_text = soup.title.string.strip() if soup.title else 'Tidak diketahui'
        nama_puskesmas = title_text.split(" - ")[-1] if " - " in title_text else title_text

        # Ambil kontak
        kontak_container = soup.find('div', class_='bg-[#34A046] w-full flex flex-col text-white space-y-1 p-2 rounded-[10px] md:space-y-2 md:p-4')
        telepon = whatsapp = 'Tidak ditemukan'
        if kontak_container:
            kontak_divs = kontak_container.find_all('div', class_='flex justify-center space-x-2 text-center font-[Epilogue] text-[10px] md:text-xs xl:text-sm')
            for div in kontak_divs:
                parts = div.find_all('div')
                if len(parts) >= 3:
                    label = parts[0].get_text(strip=True).lower()
                    nomor = parts[2].get_text(strip=True)
                    if 'telepon' in label:
                        telepon = nomor
                    elif 'whatsapp' in label:
                        whatsapp = nomor

        # Ambil semua layanan
        layanan_divs = soup.find_all('div', class_='flex fex-row w-full items-center p-4 border rounded-[6px]')
        for div in layanan_divs:
            nama_layanan_div = div.find('div', class_='w-1/2')
            nama_layanan = nama_layanan_div.get_text(strip=True) if nama_layanan_div else 'Tidak ditemukan'

            jadwal_divs = div.find('div', class_='flex flex-col items-end w-1/2')
            hari = jam = 'Tidak ditemukan'
            if jadwal_divs:
                children = jadwal_divs.find_all('div')
                if len(children) >= 2:
                    hari = children[0].get_text(strip=True)
                    jam = children[1].get_text(strip=True)

            semua_data.append({
                'nama_puskesmas': nama_puskesmas,
                'nama_layanan': nama_layanan,
                'hari': hari,
                'jam': jam,
                'telepon': telepon,
                'whatsapp': whatsapp
            })

    except Exception as e:
        print(f"❌ Gagal memproses {url}: {e}")

# Simpan ke CSV
nama_file = 'data_semua_puskesmas.csv'
if semua_data:
    with open(nama_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=semua_data[0].keys())
        writer.writeheader()
        writer.writerows(semua_data)

    print(f"✅ Data dari {len(urls)} URL berhasil disimpan ke '{nama_file}'")
else:
    print("⚠️ Tidak ada data yang berhasil diambil.")

# Tutup browser
driver.quit()
