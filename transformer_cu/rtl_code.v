// rtl_code.v -- ap_ctrl_chain wrapper around the 64-element INT8 dot product.
//
// rtl_code_core is your datapath, unchanged except for the module name.
// The wrapper adds the handshake HLS requires for an ap_ctrl_chain blackbox:
//   ap_ready    : accepts a new input every cycle the pipeline advances (II = 1)
//   ap_done     : ap_start delayed by LAT cycles, so it lines up with ap_return
//   ap_idle     : no request entering and none in flight
//   ap_continue : if a result is done but not yet accepted, the pipeline stalls

module rtl_code (
  input  wire          ap_clk,
  input  wire          ap_rst,
  input  wire          ap_ce,
  input  wire          ap_start,
  input  wire          ap_continue,
  output wire          ap_ready,
  output wire          ap_done,
  output wire          ap_idle,
  input  wire [511:0]  a,
  input  wire [511:0]  b,
  output wire [31:0]   ap_return
);
  localparam LAT = 23;              // must equal "latency" in rtl_code.json

  reg  [LAT-1:0] vld;               // tracks which pipeline slots hold a real request
  wire stall   = vld[LAT-1] & ~ap_continue;
  wire advance = ap_ce & ~stall;

  always @(posedge ap_clk) begin
    if (ap_rst)
      vld <= {LAT{1'b0}};
    else if (advance)
      vld <= {vld[LAT-2:0], ap_start};
  end

  assign ap_ready = ap_start & advance;
  assign ap_done  = vld[LAT-1];
  assign ap_idle  = ~ap_start & ~(|vld);

  rtl_code_core core (
    .ap_clk    (ap_clk),
    .ap_ce     (advance),
    .ap_rst    (ap_rst),
    .a         (a),
    .b         (b),
    .ap_return (ap_return)
  );
endmodule

module rtl_code_core (
  input wire ap_clk,
  input wire ap_ce,
  input wire ap_rst,
  input wire [512 - 1 : 0] a, 
  input wire [512 - 1 : 0] b,
  output wire [31 : 0] ap_return
);

  reg signed [7:0] ain [0:21][0:2];
  reg signed [7:0] bin [0:21][0:2];

  wire [527 : 0] ap, bp;
  assign ap = {16'b0, a};
  assign bp = {16'b0, b};

  integer i, j;

  always @ (posedge ap_clk) begin
    for (i = 0; i < 22; i = i + 1) begin
      for (j = 0; j < 3; j = j + 1) begin 
        if (ap_ce) begin
          ain[i][j] <= ap[((3 * i + j) * 8) +: 8 ];
          bin[i][j] <= bp[((3 * i + j) * 8) +: 8 ];
        end
      end
    end
  end

  genvar k;
  generate
    for (k = 0; k < 22; k = k + 1) begin : gen_k
      reg signed [31 : 0] psum;
      if (k != 0) begin
        reg signed [7 : 0] sra [0 : (k - 1)][0:2];
        reg signed [7 : 0] srb [0 : (k - 1)][0:2];
        always @(posedge ap_clk) begin
          if (ap_ce) begin : shift_regs
            integer i, m;
            for (i = 0; i < 3; i = i + 1) begin
              for (m = 0; m < (k - 1); m = m + 1) begin
                sra[m][i] <= sra[m + 1][i];
                srb[m][i] <= srb[m + 1][i];
              end
              sra[(k - 1)][i] <= ain[k][i];
              srb[(k - 1)][i] <= bin[k][i];
            end
            psum <= sra[0][0] * srb[0][0]
                  + sra[0][1] * srb[0][1]
                  + sra[0][2] * srb[0][2]
                  + gen_k[k - 1].psum;
          end
        end

      end else begin
        always @(posedge ap_clk) begin
          if (ap_ce) begin
            psum <= ain[0][0] * bin[0][0] 
                  + ain[0][1] * bin[0][1] 
                  + ain[0][2] * bin[0][2];
          end
        end
      end
    end
  endgenerate

  assign ap_return = gen_k[21].psum;
endmodule

/*
module rtl_code (
  input wire ap_clk,
  input wire ap_ce,
  input wire ap_rst,
  input wire [512 - 1 : 0] a, 
  input wire [512 - 1 : 0] b,
  output wire [31 : 0] ap_return
  //output wire          ap_ready
);
  //assign ap_ready = 1'b1;   // fully pipelined, II = 1: always ready for a new input
  reg signed [7:0] ain [0:21][0:2];
  reg signed [7:0] bin [0:21][0:2];

  wire [527 : 0] ap, bp;
  assign ap = {16'b0, a};
  assign bp = {16'b0, b};

  integer i, j;

  always @ (posedge ap_clk) begin
    for (i = 0; i < 22; i = i + 1) begin
      for (j = 0; j < 3; j = j + 1) begin 
        if (ap_ce) begin
          ain[i][j] <= ap[((3 * i + j) * 8) +: 8 ];
          bin[i][j] <= bp[((3 * i + j) * 8) +: 8 ];
        end
      end
    end
  end

  genvar k;
  generate
    for (k = 0; k < 22; k = k + 1) begin : gen_k
      reg signed [31 : 0] psum;
      if (k != 0) begin
        reg signed [7 : 0] sra [0 : (k - 1)][0:2];
        reg signed [7 : 0] srb [0 : (k - 1)][0:2];
        always @(posedge ap_clk) begin
          if (ap_ce) begin : shift_regs
            integer i, m;
            for (i = 0; i < 3; i = i + 1) begin
              for (m = 0; m < (k - 1); m = m + 1) begin
                sra[m][i] <= sra[m + 1][i];
                srb[m][i] <= srb[m + 1][i];
              end
              sra[(k - 1)][i] <= ain[k][i];
              srb[(k - 1)][i] <= bin[k][i];
            end
            psum <= sra[0][0] * srb[0][0]
                  + sra[0][1] * srb[0][1]
                  + sra[0][2] * srb[0][2]
                  + gen_k[k - 1].psum;
          end
        end

      end else begin
        always @(posedge ap_clk) begin
          if (ap_ce) begin
            psum <= ain[0][0] * bin[0][0] 
                  + ain[0][1] * bin[0][1] 
                  + ain[0][2] * bin[0][2];
          end
        end
      end
    end
  endgenerate

  assign ap_return = gen_k[21].psum;
endmodule
*/
// need acc += a * b + c * d + e * f;
// unrolls to pacc[i] = a * b + c * d + e * f + pacc[i - 1]

// first we need to split and store values as signed int8
//

