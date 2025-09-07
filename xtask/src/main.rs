use std::time::{Duration, Instant};
use rand::{seq::SliceRandom, thread_rng};
use reqwest::Client;
use hdrhistogram::Histogram;


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = std::env::args().nth(1).unwrap_or_else(|| "http://127.0.0.1:8080/v1/word".to_string());
    let clients = 8usize; // concurrent
    let total = 200usize; // total requests
    let words = vec!["communicated", "running", "happier", "analysis", "swiftly", "astonishing", "children", "better", "understand", "synthesis"];

    let client = Client::builder().pool_idle_timeout(Duration::from_secs(10)).build()?;
    let mut hist = Histogram::<u64>::new(3)?;
    let mut errors = 0usize;

    let start = Instant::now();
    let mut tasks = vec![];
    for _ in 0..clients {
        let client = client.clone();
        let url = url.clone();
        let words = words.clone();
        tasks.push(tokio::spawn(async move {
            let mut latencies = vec![];
            let mut errs = 0;
            for _ in 0..(total/clients) {
                let w = {
                    let mut rng = thread_rng();
                    words.choose(&mut rng).unwrap().to_string()
                };
                let t0 = Instant::now();
                let res = client.post(&url).json(&serde_json::json!({"word": w})).send().await;
                let dur = t0.elapsed();
                match res {
                    Ok(r) if r.status().is_success() => { latencies.push(dur); }
                    _ => errs += 1,
                }
            }
            (latencies, errs)
        }));
    }

    for t in tasks { let (ls, e) = t.await?; for d in ls { hist.record(d.as_millis() as u64).ok(); } errors += e; }

    println!("ran {} reqs in {:?}", total, start.elapsed());
    println!("errors: {}", errors);
    println!("p50: {} ms", hist.value_at_quantile(0.50));
    println!("p95: {} ms", hist.value_at_quantile(0.95));
    println!("p99: {} ms", hist.value_at_quantile(0.99));
    Ok(())
}
