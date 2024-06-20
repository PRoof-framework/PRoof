function round_2(number) {
    return Math.round(number * 100) / 100
}
function round_3(number) {
    return Math.round(number * 1000) / 1000
}

function roofline_chart(ctx, raw_data) {
    // normalize 1% time to 5 px radius
    const per_percent_px_radius = 5;
    const minimum_px_radius = 0.5;
    const maximum_px_radius = 100;
    let sum_time = 0;
    for (let i = 0; i < raw_data.length; i++) {
        sum_time += raw_data[i].time;
    }
    for (let i = 0; i < raw_data.length; i++) {
        let time_percent = raw_data[i].time / sum_time * 100
        let radius = time_percent * per_percent_px_radius;
        raw_data[i].r = Math.min(maximum_px_radius, Math.max(minimum_px_radius, radius));
    }

    const data = {
        datasets: [{
            label: 'model layers',
            data: raw_data,
        }],
    };

    const config = {
        type: 'bubble',
        data: data,
        options: {
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: "Arithmetic intensity (FLOP/byte)"
                    }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    title: {
                        display: true,
                        text: "Performance (GFLOPS)"
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => '#' + context.raw.idx + ': ' + context.raw.name
                                                + '  [' + round_2(context.raw.percentage) + '%, ' + round_3(context.raw.time*1000) + 'ms]',
                        afterLabel: (context) => round_2(context.raw.y) + " GFLOPS,  "
                                                + round_2(context.raw.x) + " FLOP/byte,  "
                                                + round_2(context.raw.memory) + " GB/s"
                    }
                }
            }
        }
    };
    const roofline_chart = new Chart(ctx, config);
    return roofline_chart;
}

function chart_set_axis(chart, type) {
    chart.config.options.scales.x.type = type;
    chart.config.options.scales.y.type = type;
    chart.update();
}
